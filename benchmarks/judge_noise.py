"""Measure the replicate noise of the DPO rubric judge, and calibrate ``filters.min_margin``.

``configs/datagen_dpo.yaml`` currently sets ``min_margin: 0.10`` on a guess. The number that
belongs there is the score gap above which the judge's ordering of two rewrites survives
being asked again: below it, a "preference pair" is a coin flip that DPO will happily fit.

This script measures three distinct quantities, which are easy to conflate:

1. **Reply determinism** — how often replicate calls return byte-identical JSON. At
   ``temperature: 0`` this may be ~100%, in which case every downstream standard deviation
   is 0 for a reason that has nothing to do with judge quality. It is reported first
   because it is the validity gate on everything after it.
2. **Per-candidate score spread** — the standard deviation of the aggregate and of each
   rubric sub-score across replicate judgings of the *same* rewrite.
3. **Pair-difference spread and sign-flip rate** — how much ``score(a) - score(b)`` moves
   across independent replicate rounds, and how often its sign flips. This is the quantity
   ``min_margin`` has to clear, and it is measured rather than derived from (2): the two
   scorings of one question share prompt context, so assuming independence would overstate
   the noise and over-set the threshold.

Candidates come from the production rewriter and are then frozen; the rewriter is not the
subject of the experiment. Judging reuses the production rubric, prompt builder, schema and
retry policy from ``data.gen_dpo_data``, so a disagreement measured here is the judge
disagreeing with itself rather than an artifact of a differently worded request.

Every individual judge sample is persisted, so the whole analysis re-runs offline from the
samples file without paying for generation again:

    PYTHONPATH=src .venv/bin/python benchmarks/judge_noise.py --config configs/datagen_dpo.yaml
    PYTHONPATH=src .venv/bin/python benchmarks/judge_noise.py --config configs/datagen_dpo.yaml \
        --from-samples data/dpo/judge_noise/samples.jsonl

Sampling is restricted to the ``train`` split by default: this is a tuning measurement, and
the test split must not inform a threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
for extra_path in (BENCHMARKS_DIR, PROJECT_ROOT / "src"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import data.settings  # noqa: E402, F401  (importing loads the project-root .env once)
from data.gen_dpo_data import (  # noqa: E402
    Candidate,
    CandidateError,
    Filters,
    RetryPolicy,
    Rubric,
    _apply_candidate_filters,
    _build_judge_messages,
    _call_with_retry,
    _read_split_qids,
    _select_within_persona_pair,
)
from data.questions import load_question  # noqa: E402
from data.settings import (  # noqa: E402
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REWRITER_API_KEY,
    REWRITER_BASE_URL,
)
from personalization.profiles import render_profile, train_personas  # noqa: E402
from rag.llm import OpenAICompatClient  # noqa: E402
from rag.rewriter import PromptedRewriter  # noqa: E402

logger = logging.getLogger(__name__)

#: Bumped whenever the meaning of a persisted sample row changes.
SAMPLES_FORMAT_VERSION = 1

#: Thresholds the tail flip-rate curve and the pair-survival curve are evaluated at. The
#: recommendation is drawn from this grid, so it is only ever as fine as the grid.
THRESHOLD_GRID = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)

#: Bin edges for reporting flip rate against observed margin.
MARGIN_BINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01)

#: Luna standard rates, USD per million tokens, for the pre-flight cost estimate only.
USD_PER_M_INPUT = 0.20
USD_PER_M_OUTPUT = 1.20

#: Rough per-call token sizes for that estimate: a judge prompt with gold context runs a
#: few hundred tokens and the schema-constrained reply is short.
EST_INPUT_TOKENS = 650
EST_OUTPUT_TOKENS = 60


# --------------------------------------------------------------------------------------
# Sampling and generation
# --------------------------------------------------------------------------------------


def _sample_question_refs(
    splits_dir: Path, split_name: str, count: int, seed: int
) -> list[tuple[str, str]]:
    """Draw *count* distinct questions from the split, reproducibly."""
    qid_pairs = _read_split_qids(splits_dir, split_name)
    if not qid_pairs:
        raise ValueError(f"Split {split_name!r} contains no question IDs")
    if count > len(qid_pairs):
        raise ValueError(
            f"Requested {count} questions but split {split_name!r} has only {len(qid_pairs)}"
        )
    return random.Random(seed).sample(sorted(qid_pairs), count)


def _generate_rewrites(
    *,
    question_refs: list[tuple[str, str]],
    questions_dir: Path,
    persona_rendered: dict[str, str],
    temperatures: list[float],
    rewriter_clients: dict[float, OpenAICompatClient],
    policy: RetryPolicy,
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate the frozen candidate set: one rewrite per (question, persona, temperature).

    Returns the candidate records and the loaded question contexts, keyed by question_ref.
    """
    contexts: dict[str, Any] = {}
    jobs: list[tuple[str, str, float]] = []
    for exam_stem, qid in question_refs:
        question_ref = f"{exam_stem}:{qid}"
        try:
            contexts[question_ref] = load_question(exam_stem, qid, questions_dir)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", question_ref, exc)
            continue
        for persona_id in persona_rendered:
            for temp in temperatures:
                jobs.append((question_ref, persona_id, temp))

    def _one(job: tuple[str, str, float]) -> dict[str, Any] | None:
        question_ref, persona_id, temp = job
        label = f"Rewrite {question_ref} persona={persona_id} temp={temp:.1f}"
        rewriter = PromptedRewriter(rewriter_clients[temp])
        query = contexts[question_ref].query

        def _once() -> str:
            rewrite = rewriter.rewrite(persona_rendered[persona_id], query)
            if not rewrite.strip():
                raise ValueError("Rewriter returned an empty completion")
            return rewrite

        try:
            rewrite = _call_with_retry(label, _once, policy)
        except CandidateError:
            logger.error("Dropping candidate — rewrite exhausted its retry budget: %s", label)
            return None
        return {
            "question_ref": question_ref,
            "persona_id": persona_id,
            "temperature": temp,
            "rewrite": rewrite,
        }

    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                candidates.append(record)

    candidates.sort(key=lambda c: (c["question_ref"], c["persona_id"], c["temperature"]))
    logger.info("Generated %d/%d candidates", len(candidates), len(jobs))
    return candidates, contexts


def _judge_replicates(
    *,
    candidates: list[dict[str, Any]],
    contexts: dict[str, Any],
    persona_rendered: dict[str, str],
    rubric: Rubric,
    judge_clients: dict[float, OpenAICompatClient],
    replicates: int,
    policy: RetryPolicy,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Judge every candidate ``replicates`` times per judge temperature.

    One row per individual call, carrying the raw reply: determinism cannot be measured
    from parsed scores alone, since two different replies can round to the same numbers.
    """
    jobs = [
        (index, candidate, judge_temp, replicate)
        for index, candidate in enumerate(candidates)
        for judge_temp in judge_clients
        for replicate in range(1, replicates + 1)
    ]

    def _one(job: tuple[int, dict[str, Any], float, int]) -> dict[str, Any]:
        index, candidate, judge_temp, replicate = job
        question_ref = candidate["question_ref"]
        persona_id = candidate["persona_id"]
        question = contexts[question_ref]
        messages = _build_judge_messages(
            rubric,
            persona_rendered[persona_id],
            question.query,
            candidate["rewrite"],
            answer=question.answer,
            explanation=question.explanation,
        )
        label = (
            f"Judge {question_ref} persona={persona_id} "
            f"temp={candidate['temperature']:.1f} jt={judge_temp:.1f} rep={replicate}"
        )
        row: dict[str, Any] = {
            "format_version": SAMPLES_FORMAT_VERSION,
            "candidate_index": index,
            "question_ref": question_ref,
            "persona_id": persona_id,
            "temperature": candidate["temperature"],
            "rewrite": candidate["rewrite"],
            "judge_temperature": judge_temp,
            "replicate": replicate,
        }
        started = time.perf_counter()
        try:
            raw = _call_with_retry(label, lambda: judge_clients[judge_temp].chat(messages), policy)
            sub_scores = rubric.parse(raw)
        except (CandidateError, ValueError) as exc:
            logger.error("Judge sample failed: %s (%s)", label, exc)
            row.update(
                raw=None,
                sub_scores=None,
                score=None,
                error=str(exc),
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return row
        row.update(
            raw=raw,
            sub_scores=sub_scores,
            score=rubric.aggregate(sub_scores),
            error=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return row

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 100 == 0:
                logger.info("Judged %d/%d samples", done, len(jobs))

    rows.sort(key=lambda r: (r["candidate_index"], r["judge_temperature"], r["replicate"]))
    return rows


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------


def _sd(values: list[float]) -> float | None:
    """Sample standard deviation, or ``None`` when a single observation makes it undefined."""
    return statistics.stdev(values) if len(values) >= 2 else None


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of an empty sequence")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _spread(values: list[float]) -> dict[str, float] | None:
    """Summarize a set of per-candidate standard deviations."""
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "max": max(values),
    }


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def _determinism(by_candidate: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """How often replicate calls came back identical, in text and in score."""
    identical_replies = 0
    identical_scores = 0
    comparable = 0
    for rows in by_candidate.values():
        usable = [r for r in rows if r["error"] is None]
        if len(usable) < 2:
            continue
        comparable += 1
        if len({r["raw"] for r in usable}) == 1:
            identical_replies += 1
        if len({round(r["score"], 6) for r in usable}) == 1:
            identical_scores += 1
    if not comparable:
        return {"candidates_with_2plus_samples": 0}
    return {
        "candidates_with_2plus_samples": comparable,
        "identical_reply_rate": identical_replies / comparable,
        "identical_score_rate": identical_scores / comparable,
    }


def _within_candidate_spread(
    by_candidate: dict[int, list[dict[str, Any]]], criteria: tuple[str, ...]
) -> dict[str, Any]:
    """Per-candidate SD of the aggregate and of every rubric sub-score."""
    collected: dict[str, list[float]] = defaultdict(list)
    for rows in by_candidate.values():
        usable = [r for r in rows if r["error"] is None]
        if len(usable) < 2:
            continue
        aggregate_sd = _sd([r["score"] for r in usable])
        if aggregate_sd is not None:
            collected["aggregate"].append(aggregate_sd)
        for name in criteria:
            criterion_sd = _sd([r["sub_scores"][name] for r in usable])
            if criterion_sd is not None:
                collected[name].append(criterion_sd)
    return {name: _spread(values) for name, values in collected.items()}


def _spread_by_score_band(by_candidate: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """SD banded by mean score: mid-range candidates are usually the noisy ones."""
    bands = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
    buckets: dict[tuple[float, float], list[float]] = {band: [] for band in bands}
    for rows in by_candidate.values():
        usable = [r for r in rows if r["error"] is None]
        if len(usable) < 2:
            continue
        mean_score = statistics.fmean([r["score"] for r in usable])
        aggregate_sd = _sd([r["score"] for r in usable])
        if aggregate_sd is None:
            continue
        for band in bands:
            if band[0] <= mean_score < band[1]:
                buckets[band].append(aggregate_sd)
                break
    return [
        {"score_from": band[0], "score_to": band[1], **(_spread(values) or {"n": 0})}
        for band, values in buckets.items()
    ]


def _pair_rounds(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build per-pair replicate rounds within each (question, persona) group.

    Round *r* pairs replicate *r* of one candidate against replicate *r* of the other, so
    each round is an independent re-run of the same comparison the generator makes once.
    """
    by_candidate_replicate: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["error"] is None:
            by_candidate_replicate[row["candidate_index"]][row["replicate"]] = row

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if index in by_candidate_replicate:
            groups[(candidate["question_ref"], candidate["persona_id"])].append(index)

    pairs: list[dict[str, Any]] = []
    for (question_ref, persona_id), members in sorted(groups.items()):
        for position, left in enumerate(sorted(members)):
            for right in sorted(members)[position + 1 :]:
                shared = sorted(
                    set(by_candidate_replicate[left]) & set(by_candidate_replicate[right])
                )
                if len(shared) < 2:
                    continue
                deltas = [
                    by_candidate_replicate[left][r]["score"]
                    - by_candidate_replicate[right][r]["score"]
                    for r in shared
                ]
                pairs.append(
                    {
                        "question_ref": question_ref,
                        "persona_id": persona_id,
                        "left": left,
                        "right": right,
                        "identical_text": candidates[left]["rewrite"].strip()
                        == candidates[right]["rewrite"].strip(),
                        "deltas": deltas,
                    }
                )
    return pairs


def _flip_table(
    pairs: list[dict[str, Any]], *, reference: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flip rate binned by observed margin, plus the tail curve used for the recommendation.

    ``reference='first'`` treats replicate 1 as the production observation and the rest as
    re-runs — the question actually being asked. ``reference='mean'`` compares each round
    against the pair's mean margin, which is the cleaner estimate of the true ordering but
    is not what a one-sample generator ever sees.
    """
    observations: list[tuple[float, bool]] = []
    for pair in pairs:
        deltas = pair["deltas"]
        if reference == "first":
            baseline = deltas[0]
            rounds = deltas[1:]
        else:
            baseline = statistics.fmean(deltas)
            rounds = deltas
        if not rounds:
            continue
        for delta in rounds:
            # A zero difference expresses no preference, so it counts as a flip: the
            # generator would have had no basis for the ordering it wrote down.
            flipped = delta == 0.0 or (delta > 0) != (baseline > 0) or baseline == 0.0
            observations.append((abs(baseline), flipped))

    binned: list[dict[str, Any]] = []
    for index in range(len(MARGIN_BINS) - 1):
        subset = [
            flipped for margin, flipped in observations if _bin_index(margin, MARGIN_BINS) == index
        ]
        binned.append(
            {
                "margin_from": MARGIN_BINS[index],
                "margin_to": MARGIN_BINS[index + 1],
                "n": len(subset),
                "flip_rate": (sum(subset) / len(subset)) if subset else None,
            }
        )

    tail: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        subset = [flipped for margin, flipped in observations if margin >= threshold]
        tail.append(
            {
                "min_margin": threshold,
                "n": len(subset),
                "flip_rate": (sum(subset) / len(subset)) if subset else None,
            }
        )
    return binned, tail


def _recommend(
    tail: list[dict[str, Any]], *, target_flip_rate: float, min_support: int
) -> dict[str, Any]:
    """Smallest grid threshold whose tail flip rate clears the target on enough evidence."""
    for entry in tail:
        if (
            entry["n"] >= min_support
            and entry["flip_rate"] is not None
            and entry["flip_rate"] < target_flip_rate
        ):
            return {
                "min_margin": entry["min_margin"],
                "flip_rate": entry["flip_rate"],
                "support": entry["n"],
                "target_flip_rate": target_flip_rate,
            }
    return {
        "min_margin": None,
        "reason": (
            "no threshold on the grid reaches the target flip rate with at least "
            f"{min_support} observations; the judge is too noisy for this grid, or the "
            "sample is too small"
        ),
        "target_flip_rate": target_flip_rate,
    }


def _survival_curve(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    filters: Filters,
    rubric: Rubric,
) -> list[dict[str, Any]]:
    """How many (question, persona) groups still yield a within-persona pair per threshold.

    Runs the production selection — candidate filters, then the upward scan past cosmetic
    near-twins — so the volume cost of raising ``min_margin`` is measured, not guessed.
    """
    means: dict[int, tuple[dict[str, float], float, int]] = {}
    for index, group in _group_by(rows, key=lambda r: r["candidate_index"]).items():
        usable = [r for r in group if r["error"] is None]
        if not usable:
            continue
        sub_scores = {
            name: statistics.fmean([r["sub_scores"][name] for r in usable])
            for name in rubric.names
        }
        means[index] = (sub_scores, rubric.aggregate(sub_scores), len(usable))

    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if index not in means:
            continue
        sub_scores, score, samples = means[index]
        grouped[(candidate["question_ref"], candidate["persona_id"])].append(
            Candidate(
                question_ref=candidate["question_ref"],
                persona_id=candidate["persona_id"],
                query="",
                rewrite=candidate["rewrite"],
                temperature=candidate["temperature"],
                sub_scores=sub_scores,
                score=score,
                judge_samples=samples,
            )
        )

    curve: list[dict[str, Any]] = []
    total = len(grouped)
    for threshold in THRESHOLD_GRID:
        tuned = replace(filters, min_margin=threshold)
        kept = 0
        for members in grouped.values():
            # _apply_candidate_filters mutates filter_reason, so hand it fresh copies.
            survivors = _apply_candidate_filters([replace(c) for c in members], tuned)
            if len(survivors) < 2:
                continue
            ranked = sorted(survivors, key=lambda c: c.score, reverse=True)
            if ranked[0].score < tuned.min_chosen_score:
                continue
            if _select_within_persona_pair(ranked, tuned) is not None:
                kept += 1
        curve.append(
            {
                "min_margin": threshold,
                "groups_with_pair": kept,
                "groups": total,
                "rate": (kept / total) if total else None,
            }
        )
    return curve


def _group_by(rows: list[dict[str, Any]], key) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def _analyze(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rubric: Rubric,
    filters: Filters,
    *,
    target_flip_rate: float,
    min_support: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for judge_temp, temp_rows in sorted(
        _group_by(rows, key=lambda r: r["judge_temperature"]).items()
    ):
        by_candidate = _group_by(temp_rows, key=lambda r: r["candidate_index"])
        failures = sum(1 for r in temp_rows if r["error"] is not None)
        pairs = _pair_rounds(temp_rows, candidates)
        deltas = [d for pair in pairs for d in pair["deltas"]]
        pair_sd = [sd for pair in pairs if (sd := _sd(pair["deltas"])) is not None]
        spread = _within_candidate_spread(by_candidate, rubric.names)
        aggregate_spread = spread.get("aggregate")
        mean_pair_sd = statistics.fmean(pair_sd) if pair_sd else None
        mean_single_sd = aggregate_spread["mean"] if aggregate_spread else None
        determinism = _determinism(by_candidate)

        binned_first, tail_first = _flip_table(pairs, reference="first")
        binned_mean, _ = _flip_table(pairs, reference="mean")

        # A fully deterministic arm has no observable flips, so the tail curve would
        # "recommend" 0.0 — an artifact of the sampler, not evidence that any margin is
        # safe. Refuse rather than emit a number the config would be tuned on.
        if determinism.get("identical_reply_rate") == 1.0:
            recommendation = {
                "min_margin": None,
                "reason": (
                    "every replicate reply was byte-identical, so replicate variance is "
                    "structurally zero and no flip can be observed. This arm measures "
                    "reproducibility, not judge uncertainty; calibrate from a "
                    "temperature > 0 arm or a prompt-perturbation pass."
                ),
                "target_flip_rate": target_flip_rate,
            }
        else:
            recommendation = _recommend(
                tail_first, target_flip_rate=target_flip_rate, min_support=min_support
            )

        report[f"{judge_temp:g}"] = {
            "n_candidates": len(by_candidate),
            "n_samples": len(temp_rows),
            "failed_samples": failures,
            "determinism": determinism,
            "within_candidate_sd": spread,
            "sd_by_score_band": _spread_by_score_band(by_candidate),
            "pairs": {
                "n_pairs": len(pairs),
                "n_identical_text_pairs": sum(1 for p in pairs if p["identical_text"]),
                "mean_abs_delta": statistics.fmean([abs(d) for d in deltas]) if deltas else None,
                "mean_pair_delta_sd": mean_pair_sd,
                "independent_prediction": (
                    mean_single_sd * math.sqrt(2) if mean_single_sd is not None else None
                ),
                "implied_correlation": (
                    1 - (mean_pair_sd**2) / (2 * mean_single_sd**2)
                    if mean_pair_sd is not None and mean_single_sd
                    else None
                ),
                "flip_by_margin_vs_first": binned_first,
                "flip_by_margin_vs_mean": binned_mean,
                "tail_flip_curve": tail_first,
                "recommendation": recommendation,
            },
            "pair_survival": _survival_curve(temp_rows, candidates, filters, rubric),
        }
    return report


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_report(report: dict[str, Any], current_min_margin: float) -> None:
    for judge_temp, section in report.items():
        print(f"\n{'=' * 78}\njudge temperature {judge_temp}\n{'=' * 78}")
        print(
            f"candidates {section['n_candidates']}  samples {section['n_samples']}  "
            f"failed {section['failed_samples']}"
        )

        determinism = section["determinism"]
        if determinism.get("candidates_with_2plus_samples"):
            print(
                f"determinism: identical replies {_fmt(determinism['identical_reply_rate'], 3)}  "
                f"identical scores {_fmt(determinism['identical_score_rate'], 3)}"
            )
            if determinism["identical_reply_rate"] == 1.0:
                print(
                    "  ! every replicate was byte-identical: this arm measures reproducibility, "
                    "not judge uncertainty. Read the higher-temperature arm, or perturb the "
                    "prompt instead of the sampler."
                )
        else:
            print("determinism: not measurable (fewer than 2 usable samples per candidate)")

        print("\nwithin-candidate SD")
        for name, spread in section["within_candidate_sd"].items():
            if spread:
                print(
                    f"  {name:<22} mean {_fmt(spread['mean'])}  p50 {_fmt(spread['p50'])}  "
                    f"p90 {_fmt(spread['p90'])}  max {_fmt(spread['max'])}  n={spread['n']}"
                )

        print("\nSD by score band")
        for band in section["sd_by_score_band"]:
            if band.get("n"):
                print(
                    f"  [{band['score_from']:.2f}, {band['score_to']:.2f})  "
                    f"mean {_fmt(band['mean'])}  n={band['n']}"
                )

        pairs = section["pairs"]
        print(
            f"\npairs {pairs['n_pairs']} "
            f"(identical text: {pairs['n_identical_text_pairs']})  "
            f"mean |delta| {_fmt(pairs['mean_abs_delta'])}"
        )
        print(
            f"  pair-difference SD {_fmt(pairs['mean_pair_delta_sd'])}  "
            f"independent prediction {_fmt(pairs['independent_prediction'])}  "
            f"implied correlation {_fmt(pairs['implied_correlation'], 3)}"
        )

        print("\nflip rate by observed margin (replicate 1 as the observation)")
        for entry in pairs["flip_by_margin_vs_first"]:
            print(
                f"  [{entry['margin_from']:.2f}, {entry['margin_to']:.2f})  "
                f"n={entry['n']:<6} flip {_fmt(entry['flip_rate'], 3)}"
            )

        print("\ntail flip rate and pair yield by threshold")
        survival = {row["min_margin"]: row for row in section["pair_survival"]}
        for entry in pairs["tail_flip_curve"]:
            row = survival.get(entry["min_margin"], {})
            marker = "  <- current" if entry["min_margin"] == current_min_margin else ""
            print(
                f"  min_margin {entry['min_margin']:.2f}  n={entry['n']:<6} "
                f"flip {_fmt(entry['flip_rate'], 3)}  "
                f"groups with pair {row.get('groups_with_pair', 'n/a')}/"
                f"{row.get('groups', 'n/a')}{marker}"
            )

        recommendation = pairs["recommendation"]
        if recommendation.get("min_margin") is not None:
            print(
                f"\nrecommended filters.min_margin: {recommendation['min_margin']:.2f} "
                f"(flip {_fmt(recommendation['flip_rate'], 3)} over "
                f"{recommendation['support']} observations, target "
                f"{recommendation['target_flip_rate']:.2f})"
            )
        else:
            print(f"\nno recommendation: {recommendation['reason']}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _parse_temps(raw: str) -> list[float]:
    temps = [float(part) for part in raw.split(",") if part.strip()]
    if not temps:
        raise argparse.ArgumentTypeError("at least one judge temperature is required")
    return temps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure DPO judge replicate noise and calibrate filters.min_margin"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the datagen config")
    parser.add_argument("--split", default="train", help="Split to sample questions from")
    parser.add_argument("--questions", type=int, default=25, help="Number of questions to sample")
    parser.add_argument(
        "--replicates", type=int, default=5, help="Judge calls per candidate per temperature"
    )
    parser.add_argument(
        "--judge-temps",
        type=_parse_temps,
        default=[0.0, 1.0],
        help=(
            "Comma-separated judge temperatures. 0.0 is the production setting; 1.0 is the "
            "retired v1 setting and the fallback when 0.0 is fully deterministic."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Question-sampling seed")
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=Path("data/dpo/judge_noise/samples.jsonl"),
        help="Where every individual judge sample is written",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/dpo/judge_noise/report.json"),
        help="Where the analysis is written",
    )
    parser.add_argument(
        "--from-samples",
        type=Path,
        default=None,
        help="Re-analyze an existing samples file without making any API call",
    )
    parser.add_argument(
        "--target-flip-rate",
        type=float,
        default=0.05,
        help="Sign-flip rate the recommended min_margin must stay under",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum observations behind a recommended threshold",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Override config workers")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the call budget and cost estimate, then exit"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rewriter_cfg = config["rewriter"]
    judge_cfg = config["judge"]
    data_cfg = config["data"]
    rubric = Rubric.from_config(judge_cfg["rubric"])
    filters = Filters.from_config(config["filters"])
    temperatures: list[float] = rewriter_cfg["temperatures"]
    max_workers = args.max_workers or rewriter_cfg.get("max_workers", 4)

    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2: noise needs two observations")

    if args.from_samples is not None:
        rows = [
            json.loads(line)
            for line in args.from_samples.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stale = {r.get("format_version") for r in rows} - {SAMPLES_FORMAT_VERSION}
        if stale:
            raise ValueError(
                f"{args.from_samples} has sample format versions {sorted(stale)}, "
                f"expected {SAMPLES_FORMAT_VERSION}"
            )
        candidates: list[dict[str, Any]] = []
        seen: dict[int, dict[str, Any]] = {}
        for row in rows:
            seen.setdefault(
                row["candidate_index"],
                {
                    "question_ref": row["question_ref"],
                    "persona_id": row["persona_id"],
                    "temperature": row["temperature"],
                    "rewrite": row["rewrite"],
                },
            )
        candidates = [seen[index] for index in sorted(seen)]
        # candidate_index is the position in this list, which the sort above preserves.
        logger.info("Loaded %d samples over %d candidates", len(rows), len(candidates))
    else:
        persona_ids = [p.id for p in train_personas()]
        persona_rendered = {pid: render_profile(pid) for pid in persona_ids}
        question_refs = _sample_question_refs(
            Path(data_cfg["splits_dir"]), args.split, args.questions, args.seed
        )

        n_candidates = len(question_refs) * len(persona_ids) * len(temperatures)
        n_judge_calls = n_candidates * len(args.judge_temps) * args.replicates
        cost = (
            n_judge_calls * EST_INPUT_TOKENS * USD_PER_M_INPUT
            + n_judge_calls * EST_OUTPUT_TOKENS * USD_PER_M_OUTPUT
        ) / 1_000_000
        logger.info(
            "Budget: %d rewrites, %d judge calls (%d candidates x %d temps x %d replicates), "
            "judge cost approx $%.2f",
            n_candidates,
            n_judge_calls,
            n_candidates,
            len(args.judge_temps),
            args.replicates,
            cost,
        )
        if args.dry_run:
            return

        rewriter_clients = {
            temp: OpenAICompatClient(
                base_url=REWRITER_BASE_URL,
                api_key=REWRITER_API_KEY,
                model=rewriter_cfg["model"],
                temperature=temp,
                max_tokens=rewriter_cfg["max_completion_tokens"],
            )
            for temp in temperatures
        }
        judge_clients = {
            judge_temp: OpenAICompatClient(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                model=judge_cfg["model"],
                temperature=judge_temp,
                max_tokens=judge_cfg["max_completion_tokens"],
                reasoning_effort=judge_cfg.get("reasoning_effort"),
                response_format=rubric.response_format(),
            )
            for judge_temp in args.judge_temps
        }

        candidates, contexts = _generate_rewrites(
            question_refs=question_refs,
            questions_dir=Path(data_cfg["questions_dir"]),
            persona_rendered=persona_rendered,
            temperatures=temperatures,
            rewriter_clients=rewriter_clients,
            policy=RetryPolicy.from_config(rewriter_cfg, "rewriter"),
            max_workers=max_workers,
        )
        if not candidates:
            logger.error("No candidates were generated; nothing to judge")
            return

        rows = _judge_replicates(
            candidates=candidates,
            contexts=contexts,
            persona_rendered=persona_rendered,
            rubric=rubric,
            judge_clients=judge_clients,
            replicates=args.replicates,
            policy=RetryPolicy.from_config(judge_cfg, "judge"),
            max_workers=max_workers,
        )

        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples to %s", len(rows), args.samples_out)

    analysis = _analyze(
        rows,
        candidates,
        rubric,
        filters,
        target_flip_rate=args.target_flip_rate,
        min_support=args.min_support,
    )
    report = {
        "format_version": SAMPLES_FORMAT_VERSION,
        "config": str(args.config),
        "split": args.split,
        "seed": args.seed,
        "replicates": args.replicates,
        "rewriter_model": rewriter_cfg["model"],
        "rewriter_temperatures": temperatures,
        "judge_model": judge_cfg["model"],
        "judge_reasoning_effort": judge_cfg.get("reasoning_effort"),
        "rubric": [{"name": c.name, "weight": c.weight} for c in rubric.criteria],
        "current_min_margin": filters.min_margin,
        "target_flip_rate": args.target_flip_rate,
        "by_judge_temperature": analysis,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_report(analysis, filters.min_margin)
    print(f"\nreport written to {args.report_out}")


if __name__ == "__main__":
    main()
