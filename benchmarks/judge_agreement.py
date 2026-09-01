"""Measure how often the tournament judge agrees with the labeller that built the pairs.

The preference pairs in ``data/dpo/`` were labelled by ``gpt-5.6-luna``; the arm tournament
in ``compare_dpo_rewriters.py`` is judged by a Gemini model. If the two disagree about what
a good rewrite is, then the training target and the evaluation target are different
quantities and no objective or checkpoint rule can win. This script measures that agreement
directly on a random sample of the existing pairs.

Every judged pair reuses the tournament's own system prompt, response schema, gold-context
block and retry policy, so a disagreement measured here is a disagreement about rewrites
rather than an artifact of a differently worded request.

Run it locally:

    uv run python benchmarks/judge_agreement.py --sample 100

Reads ``DPO_EVAL_BASE_URL``, ``DPO_EVAL_API_KEY`` and ``DPO_EVAL_MODEL`` from the
environment or a project-root ``.env``.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
for extra_path in (BENCHMARKS_DIR, PROJECT_ROOT / "src"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import data.settings  # noqa: E402, F401  (importing loads the project-root .env once)
from compare_dpo_rewriters import (  # noqa: E402
    _JUDGE_SCHEMA,
    _JUDGE_SYSTEM,
    JudgeUnavailable,
    PromptRecord,
    RetryPolicy,
    _judge_one,
    _parse_judgment,
    _parse_question_ref,
)
from data.questions import load_question, render_question_value  # noqa: E402
from personalization.profiles import render_profile  # noqa: E402
from rag.llm import GeminiClient  # noqa: E402
from rl.dpo_train import load_pairs  # noqa: E402

logger = logging.getLogger(__name__)


def _sample_pairs(
    pairs: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Draw one pair per question/persona key so no question is counted twice.

    A question/persona key appears in several rows of the training split, and rows sharing a
    key share a question. Sampling rows directly would let one question dominate the sample
    and would break the independence the binomial interval below assumes.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        if pair["chosen"].strip() == pair["rejected"].strip():
            continue
        grouped.setdefault((pair["question_ref"], pair["persona_id"]), []).append(pair)
    keys = sorted(grouped)
    if sample_size > len(keys):
        raise ValueError(f"Requested {sample_size} pairs but only {len(keys)} keys are available")
    rng = random.Random(seed)
    selected = rng.sample(keys, sample_size)
    return [rng.choice(grouped[key]) for key in selected]


def _build_jobs(
    sample: list[dict[str, Any]], questions_dir: Path
) -> tuple[list[dict[str, Any]], list[PromptRecord], dict[str, dict[tuple[str, str], str]]]:
    """Build balanced-orientation jobs: the chosen rewrite is A in exactly half of them."""
    prompts: list[PromptRecord] = []
    chosen_by_key: dict[tuple[str, str], str] = {}
    rejected_by_key: dict[tuple[str, str], str] = {}
    jobs: list[dict[str, Any]] = []
    for index, pair in enumerate(sample):
        exam_stem, question_id = _parse_question_ref(pair["question_ref"])
        question = load_question(exam_stem, question_id, questions_dir)
        if question.query != pair["query"]:
            raise ValueError(f"Pair query for {pair['question_ref']} differs from data/questions")
        key = pair["question_ref"], pair["persona_id"]
        prompts.append(
            PromptRecord(
                question_ref=pair["question_ref"],
                persona_id=pair["persona_id"],
                query=pair["query"],
                profile=render_profile(pair["persona_id"]),
                answer="" if question.answer is None else render_question_value(question.answer),
                explanation=question.explanation or "",
            )
        )
        chosen_by_key[key] = pair["chosen"]
        rejected_by_key[key] = pair["rejected"]
        chosen_is_a = index % 2 == 0
        jobs.append(
            {
                "question_ref": pair["question_ref"],
                "persona_id": pair["persona_id"],
                "candidate_a": "chosen" if chosen_is_a else "rejected",
                "candidate_b": "rejected" if chosen_is_a else "chosen",
                "chosen_position": "A" if chosen_is_a else "B",
            }
        )
    outputs = {"chosen": chosen_by_key, "rejected": rejected_by_key}
    return jobs, prompts, outputs


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at the extremes where the normal interval does not."""
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
    ) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial test against p=0.5, summing tails no likelier than observed."""
    if trials == 0:
        return float("nan")
    observed = math.comb(trials, successes)
    total = sum(
        math.comb(trials, count) for count in range(trials + 1) if math.comb(trials, count) <= observed
    )
    return min(1.0, total / 2**trials)


def _check_endpoint_values(base_url: str, model: str) -> None:
    """Reject endpoint values that cannot work, before spending the sample on them.

    A comma in either value almost always means the shell was handed
    ``VAR=a, VAR=b`` instead of ``VAR=a VAR=b``, which silently appends the comma to the
    first value. That produces an unresolvable host and a connection error carrying no
    status code, which the retry policy cannot distinguish from a network blip.
    """
    for name, value in (("DPO_EVAL_BASE_URL", base_url), ("DPO_EVAL_MODEL", model)):
        if value != value.strip():
            raise ValueError(f"{name} has surrounding whitespace: {value!r}")
        if "," in value:
            raise ValueError(
                f"{name} contains a comma: {value!r}. Separate environment assignments with "
                "spaces, not commas: VAR=a VAR=b, never VAR=a, VAR=b."
            )
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"DPO_EVAL_BASE_URL is not an http(s) URL with a host: {base_url!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"DPO_EVAL_BASE_URL must be a bare host, got {base_url!r}")


def _preflight_judge(client: GeminiClient, model: str, base_url: str) -> None:
    """Spend one request proving the endpoint answers, so a misconfiguration fails fast.

    Without this, every unreachable-host failure looks transient, each job burns its retries,
    and the run ends with a full set of `missing` rows and an all-NaN report.
    """
    probe = (
        "Learner profile:\nprobe\n\nOriginal complete question:\nprobe\n\n"
        "Judge-only gold context:\nNo gold explanation supplied.\n\n"
        "Rewrite A:\nalpha\n\nRewrite B:\nbeta"
    )
    try:
        _parse_judgment(client.generate(probe))
    except Exception as exc:
        raise RuntimeError(
            f"Judge endpoint is not usable, so no pair was judged.\n"
            f"  base_url: {base_url!r}\n"
            f"  model:    {model!r}\n"
            f"  error:    {type(exc).__name__}: {exc}"
        ) from exc


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn per-pair verdicts into agreement rates, an interval, and a bias check."""
    judged = [record for record in results if record["status"] == "ok"]
    agree = [r for r in judged if r["winner"] == r["chosen_position"]]
    tie = [r for r in judged if r["winner"] == "TIE"]
    disagree = [r for r in judged if r["winner"] not in {"TIE", r["chosen_position"]}]
    decisive = len(agree) + len(disagree)
    picked_a = sum(1 for r in judged if r["winner"] == "A")
    picked_b = sum(1 for r in judged if r["winner"] == "B")

    by_persona: dict[str, dict[str, int]] = {}
    for record in judged:
        counts = by_persona.setdefault(
            record["persona_id"], {"agree": 0, "disagree": 0, "tie": 0}
        )
        if record["winner"] == "TIE":
            counts["tie"] += 1
        elif record["winner"] == record["chosen_position"]:
            counts["agree"] += 1
        else:
            counts["disagree"] += 1

    # A missing row carries the last error it saw. Aggregate them: a run that judged nothing
    # must say why in the report, not only in the per-row file.
    errors = collections.Counter(
        record.get("error") or "unknown" for record in results if record["status"] != "ok"
    )
    low, high = _wilson_interval(len(agree), decisive)
    return {
        "sample": len(results),
        "judged": len(judged),
        "missing": len(results) - len(judged),
        "errors": dict(errors.most_common()),
        "agree": len(agree),
        "disagree": len(disagree),
        "tie": len(tie),
        # Ties carry no information about who is right, so the headline rate excludes them
        # and the tie rate is reported beside it.
        "decisive": decisive,
        "agreement_decisive": (len(agree) / decisive) if decisive else float("nan"),
        "agreement_ci95": [low, high],
        "agreement_p_vs_chance": _binomial_two_sided_p(len(agree), decisive),
        "tie_rate": (len(tie) / len(judged)) if judged else float("nan"),
        # Position bias: with balanced orientation an unbiased judge splits A and B evenly.
        "picked_a": picked_a,
        "picked_b": picked_b,
        "position_bias_p": _binomial_two_sided_p(picked_a, picked_a + picked_b),
        "per_persona": {
            persona: {
                **counts,
                "agreement_decisive": (
                    counts["agree"] / (counts["agree"] + counts["disagree"])
                    if counts["agree"] + counts["disagree"]
                    else float("nan")
                ),
            }
            for persona, counts in sorted(by_persona.items())
        },
    }


def _report(summary: dict[str, Any], model: str) -> str:
    rate = summary["agreement_decisive"]
    low, high = summary["agreement_ci95"]
    lines = [
        "",
        f"Judge:            {model}",
        f"Pairs sampled:    {summary['sample']}  (judged {summary['judged']}, "
        f"missing {summary['missing']})",
        f"Agree / disagree: {summary['agree']} / {summary['disagree']}"
        f"   ties {summary['tie']} ({summary['tie_rate']:.1%})",
        f"Agreement:        {rate:.1%}  95% CI [{low:.1%}, {high:.1%}]"
        f"   p vs chance {summary['agreement_p_vs_chance']:.4g}",
        f"Position split:   A {summary['picked_a']} / B {summary['picked_b']}"
        f"   p vs even {summary['position_bias_p']:.4g}",
        "",
        "Per persona (decisive verdicts only):",
    ]
    for persona, counts in summary["per_persona"].items():
        lines.append(
            f"  {persona:10} agree {counts['agree']:3d}  disagree {counts['disagree']:3d}"
            f"  ties {counts['tie']:3d}  -> {counts['agreement_decisive']:.1%}"
        )
    if summary["errors"]:
        lines.append("")
        lines.append("Failures by last error:")
        for message, count in summary["errors"].items():
            lines.append(f"  {count:4d}  {message[:150]}")
    lines.append("")
    if math.isnan(rate):
        lines.append(
            "No decisive verdict, so nothing can be concluded about agreement. The errors "
            "above are the reason; fix the endpoint or the sample, then rerun."
        )
    elif high < 0.6:
        lines.append(
            "Agreement is near chance. The labeller and the tournament judge are measuring "
            "different things, so no training objective can improve the judged score."
        )
    elif low > 0.75:
        lines.append(
            "The two judges broadly agree. Poor tournament scores are a training problem, "
            "not a label/metric mismatch."
        )
    else:
        lines.append(
            "Agreement is partial. Some of the ceiling on judged quality comes from label "
            "noise; a larger sample would tighten this interval."
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_dpo.yaml")
    parser.add_argument(
        "--pairs", type=Path, help="Preference pairs to sample (default: config data.train_path)"
    )
    parser.add_argument("--sample", type=int, default=100, help="Pairs to judge (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "rl" / "agreement")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pairs_path = args.pairs or Path(config["data"]["train_path"])
    if args.sample < 1:
        raise ValueError("--sample must be at least 1")

    base_url = os.environ.get("DPO_EVAL_BASE_URL")
    api_key = os.environ.get("DPO_EVAL_API_KEY")
    model = os.environ.get("DPO_EVAL_MODEL")
    missing_env = [
        name
        for name, value in (
            ("DPO_EVAL_BASE_URL", base_url),
            ("DPO_EVAL_API_KEY", api_key),
            ("DPO_EVAL_MODEL", model),
        )
        if not value
    ]
    if missing_env:
        raise RuntimeError("Missing required judge environment: " + ", ".join(missing_env))
    if "gemini" not in model.lower():
        raise ValueError(
            "DPO_EVAL_MODEL must identify the same Gemini-family judge as the tournament"
        )
    _check_endpoint_values(base_url, model)

    sample = _sample_pairs(load_pairs(pairs_path), args.sample, args.seed)
    jobs, prompts, outputs = _build_jobs(sample, Path(config["data"]["questions_dir"]))
    prompts_by_key = {prompt.key: prompt for prompt in prompts}
    comparison = config["comparison"]
    retry = RetryPolicy.from_config(comparison["retry"])
    client = GeminiClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_instruction=_JUDGE_SYSTEM,
        temperature=0.0,
        max_output_tokens=int(comparison["judge_max_tokens"]),
        thinking_level=comparison.get("judge_thinking_level", "minimal"),
        response_schema=_JUDGE_SCHEMA,
    )
    _preflight_judge(client, model, base_url)

    results: list[dict[str, Any]] = []
    fatal: JudgeUnavailable | None = None
    with ThreadPoolExecutor(max_workers=int(comparison["max_workers"])) as executor:
        futures = {
            executor.submit(_judge_one, job, prompts_by_key, outputs, client, retry): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except JudgeUnavailable as exc:
                # The same rejection would repeat for every remaining pair.
                fatal = exc
                executor.shutdown(wait=False, cancel_futures=True)
                break
            logger.info("Judged %d/%d", index, len(jobs))
    if fatal is not None:
        raise fatal

    summary = summarize(results)
    summary["inputs"] = {
        "pairs_path": str(pairs_path),
        "sample_size": args.sample,
        "seed": args.seed,
        "judge_model": model,
        "judge_thinking_level": comparison.get("judge_thinking_level", "minimal"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pairs_path.stem}-n{args.sample}-seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(results, key=lambda r: (r["question_ref"], r["persona_id"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(_report(summary, model))
    print(f"Wrote {args.output_dir / f'{stem}.json'}")
    if summary["decisive"] == 0:
        # An all-NaN report is not a result. Exit nonzero so a caller cannot mistake it
        # for a measured agreement of zero.
        sys.exit(1)


if __name__ == "__main__":
    main()
