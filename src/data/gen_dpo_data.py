"""Generate DPO rewriter preference pairs: N rewrites per (query, persona) ranked by LLM judge.

The judge scores a rubric, not a single gestalt number: it returns one sub-score per
criterion under a strict JSON schema, those sub-scores are averaged over several judge
samples, and the configured weights collapse them into one aggregate score. Every
candidate and every score is persisted, so pair filtering can be re-tuned from the
candidate dumps without paying for generation again.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from tqdm import tqdm

from data.questions import QuestionContext, load_question, render_question_value
from data.remove_degenerate_dpo_pairs import normalize_rewrite
from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITER_API_KEY, REWRITER_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.llm import OpenAICompatClient
from rag.rewriter import PromptedRewriter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are an expert Persian language tutor evaluating query rewrites for a RAG retrieval system."

# Bumped whenever the meaning of a written row changes.
#
# Version 1: ``query`` became the complete rendered question (data.questions.load_question)
# rather than the bare stem, and scores came from a judge that saw the gold answer.
# Version 2: the judge scores a weighted rubric under a strict JSON schema at a fixed
# temperature, averaged over several samples; rows carry the judge scores, the pair type,
# and the persona each completion was written for, so filtering and analysis no longer
# need provenance the file never stored.
DPO_OUTPUT_FORMAT_VERSION = 2

# Candidate dumps version independently of pair rows: they are analysis inputs, not
# training inputs, and ``rl.dpo_train`` never reads them.
DPO_CANDIDATE_FORMAT_VERSION = 1

WITHIN_PERSONA = "within_persona"
CROSS_PERSONA = "cross_persona"

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential, capped backoff budget for one API call site."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0

    @classmethod
    def from_config(cls, block: dict, block_name: str) -> RetryPolicy:
        policy = cls(
            max_attempts=block.get("max_attempts", 3),
            initial_backoff_seconds=block.get("initial_backoff_seconds", 1.0),
            backoff_multiplier=block.get("backoff_multiplier", 2.0),
            max_backoff_seconds=block.get("max_backoff_seconds", 8.0),
        )
        policy.validate(block_name)
        return policy

    def validate(self, block_name: str) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError(f"{block_name}.max_attempts must be an integer >= 1")
        for field_name, value in (
            ("initial_backoff_seconds", self.initial_backoff_seconds),
            ("backoff_multiplier", self.backoff_multiplier),
            ("max_backoff_seconds", self.max_backoff_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{block_name}.{field_name} must be a finite number")
        if self.initial_backoff_seconds < 0:
            raise ValueError(f"{block_name}.initial_backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError(f"{block_name}.backoff_multiplier must be >= 1")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                f"{block_name}.max_backoff_seconds must be >= {block_name}.initial_backoff_seconds"
            )


def _require_unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class RubricCriterion:
    """One scored axis of the judge rubric, with the weight it carries in the aggregate."""

    name: str
    weight: float
    description: str


@dataclass(frozen=True)
class Rubric:
    """The full weighted rubric: prompt text, response schema, and aggregation live here."""

    criteria: tuple[RubricCriterion, ...]

    @classmethod
    def from_config(cls, block: object) -> Rubric:
        if not isinstance(block, list) or not block:
            raise ValueError("judge.rubric must be a nonempty list of criteria")
        criteria: list[RubricCriterion] = []
        seen: set[str] = set()
        for index, entry in enumerate(block):
            if not isinstance(entry, dict):
                raise ValueError(f"judge.rubric[{index}] must be a mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"judge.rubric[{index}].name must be a nonempty string")
            if name in seen:
                raise ValueError(f"judge.rubric has duplicate criterion {name!r}")
            seen.add(name)
            description = entry.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"judge.rubric[{index}].description must be a nonempty string")
            weight = entry.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"judge.rubric[{index}].weight must be a number")
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"judge.rubric[{index}].weight must be a positive finite number")
            criteria.append(RubricCriterion(name=name, weight=weight, description=description))

        total = math.fsum(criterion.weight for criterion in criteria)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"judge.rubric weights must sum to 1.0, got {total:.6f}")
        return cls(criteria=tuple(criteria))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(criterion.name for criterion in self.criteria)

    def prompt_block(self) -> str:
        """Render the rubric as the instruction block the judge reads."""
        return "\n".join(
            f"  - {criterion.name} (weight {criterion.weight:g}): {criterion.description}"
            for criterion in self.criteria
        )

    def response_format(self) -> dict:
        """Strict JSON schema constraining the judge to exactly these sub-scores."""
        properties = {
            criterion.name: {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": criterion.description,
            }
            for criterion in self.criteria
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "rewrite_rubric",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(self.names),
                    "additionalProperties": False,
                },
            },
        }

    def parse(self, response: str) -> dict[str, float]:
        """Parse one judge reply into a complete set of sub-scores in ``[0, 1]``."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Judge response is empty: {response!r}")
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Judge response is not JSON: {response!r}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Judge response is not a JSON object: {response!r}")
        return {
            criterion.name: _require_unit_float(
                payload.get(criterion.name), f"judge sub-score {criterion.name!r}"
            )
            for criterion in self.criteria
        }

    def aggregate(self, sub_scores: dict[str, float]) -> float:
        """Collapse sub-scores into the single number pair selection orders by."""
        return math.fsum(
            criterion.weight * sub_scores[criterion.name] for criterion in self.criteria
        )


@dataclass(frozen=True)
class Filters:
    """Config-driven filters. Candidate filters gate the dump; pair filters gate training rows."""

    #: Candidate filter: a candidate scored by fewer than this many successful judge
    #: samples carries more judge noise than the rest and is not comparable to them.
    min_judge_samples: int
    #: Candidate filter: collapse candidates that normalize to the same string, keeping
    #: the highest-scored one. Identical text cannot express a preference.
    dedup_candidates: bool
    #: Pair filter: the chosen completion must clear this absolute quality bar.
    min_chosen_score: float
    #: Pair filter: the score gap must exceed the judge's own noise floor, otherwise the
    #: pair teaches a coin flip. This is the knob that replaces "best minus worst, always".
    min_margin: float
    #: Pair filter: reject pairs whose completions are near-identical strings, where a
    #: score gap reflects judge jitter rather than a real difference.
    max_pair_similarity: float
    #: Pair filter: the same margin rule for cross-persona rows, which are scored under
    #: the *target* persona so both sides are directly comparable.
    cross_persona_min_margin: float

    @classmethod
    def from_config(cls, block: object) -> Filters:
        if not isinstance(block, dict):
            raise ValueError("filters must be a mapping")
        min_judge_samples = block.get("min_judge_samples", 1)
        if (
            isinstance(min_judge_samples, bool)
            or not isinstance(min_judge_samples, int)
            or min_judge_samples < 1
        ):
            raise ValueError("filters.min_judge_samples must be an integer >= 1")
        dedup_candidates = block.get("dedup_candidates", True)
        if not isinstance(dedup_candidates, bool):
            raise ValueError("filters.dedup_candidates must be a boolean")
        return cls(
            min_judge_samples=min_judge_samples,
            dedup_candidates=dedup_candidates,
            min_chosen_score=_require_unit_float(
                block.get("min_chosen_score", 0.0), "filters.min_chosen_score"
            ),
            min_margin=_require_unit_float(block.get("min_margin", 0.0), "filters.min_margin"),
            max_pair_similarity=_require_unit_float(
                block.get("max_pair_similarity", 1.0), "filters.max_pair_similarity"
            ),
            cross_persona_min_margin=_require_unit_float(
                block.get("cross_persona_min_margin", 0.0), "filters.cross_persona_min_margin"
            ),
        )


@dataclass
class Candidate:
    """One rewrite plus the judge verdict on it, as persisted to the candidate dumps."""

    question_ref: str
    persona_id: str
    query: str
    rewrite: str
    temperature: float
    sub_scores: dict[str, float]
    score: float
    judge_samples: int
    #: ``None`` while the candidate is live; set to the filter name that discarded it.
    filter_reason: str | None = None
    #: Cross-persona scores of this rewrite under other personas, filled in only for the
    #: per-persona winners. Keyed by the persona the rewrite was scored *as*.
    scored_as: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        record = {
            "format_version": DPO_CANDIDATE_FORMAT_VERSION,
            "question_ref": self.question_ref,
            "persona_id": self.persona_id,
            "query": self.query,
            "rewrite": self.rewrite,
            "temperature": self.temperature,
            "sub_scores": self.sub_scores,
            "score": self.score,
            "judge_samples": self.judge_samples,
            "filter_reason": self.filter_reason,
            "scored_as": self.scored_as,
        }
        return json.dumps(record, ensure_ascii=False) + "\n"


class CandidateError(RuntimeError):
    """Raised when a rewrite or its score cannot be produced within the retry budget."""


def _call_with_retry(label: str, call: Callable[[], T], policy: RetryPolicy) -> T:
    backoff_seconds = policy.initial_backoff_seconds
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            logger.warning(
                "%s failed (attempt %d/%d)", label, attempt, policy.max_attempts, exc_info=True
            )
            if attempt == policy.max_attempts:
                raise CandidateError(
                    f"{label} failed after {policy.max_attempts} attempts"
                ) from exc
            time.sleep(backoff_seconds)
            backoff_seconds = min(
                backoff_seconds * policy.backoff_multiplier, policy.max_backoff_seconds
            )
    raise CandidateError(f"{label} failed")


def _build_judge_messages(
    rubric: Rubric,
    persona_rendered: str,
    original_query: str,
    rewrite: str,
    answer: object | None = None,
    explanation: str | None = None,
) -> list[dict[str, str]]:
    # Gold fields are judge context only. They tell the judge which material actually
    # resolves the question, and they never reach the rewriter — a rewrite conditioned on
    # the answer would leak it into the retrieval query. Same split as gen_ropg_data.
    reference_sections: list[str] = []
    if answer is not None:
        rendered_answer = answer if isinstance(answer, str) else render_question_value(answer)
        reference_sections.append(f"Gold answer/reference:\n{rendered_answer}")
    if explanation is not None:
        reference_sections.append(f"Gold explanation/rubric:\n{explanation}")
    reference_context = ""
    if reference_sections:
        reference_context = (
            "Reference answer and rubric (judge context only; the rewriter never saw this):\n"
            + "\n\n".join(reference_sections)
            + "\n\n"
        )

    user = (
        f"A student with the following profile is searching for study material:\n"
        f"Profile: {persona_rendered}\n\n"
        f"Original exam question: {original_query}\n\n"
        f"{reference_context}"
        f"Candidate rewrite: {rewrite}\n\n"
        "Score how well this rewrite would help retrieve the right study material for this "
        "specific student. Rate each criterion independently from 0.0 to 1.0, where 0.0 is a "
        "complete failure on that criterion and 1.0 could not be improved:\n"
        f"{rubric.prompt_block()}\n\n"
        "Judge each criterion on its own merits — a rewrite may score high on one and low on "
        "another. Use the full range; reserve scores above 0.9 for rewrites you cannot improve."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _score_rewrite(
    judge_client: OpenAICompatClient,
    rubric: Rubric,
    messages: list[dict[str, str]],
    samples: int,
    policy: RetryPolicy,
    label: str,
) -> tuple[dict[str, float], float, int]:
    """Score one rewrite ``samples`` times and average the sub-scores.

    Returns ``(mean_sub_scores, aggregate, successful_samples)``. Averaging happens on the
    sub-scores, not on the aggregate: the two agree because the aggregate is linear in the
    sub-scores, and per-criterion means are what the candidate dumps record.
    """
    totals: dict[str, float] = defaultdict(float)
    collected = 0
    for sample_index in range(1, samples + 1):
        sample_label = f"{label} sample {sample_index}/{samples}"

        def _score_once() -> dict[str, float]:
            return rubric.parse(judge_client.chat(messages))

        try:
            sub_scores = _call_with_retry(sample_label, _score_once, policy)
        except CandidateError:
            logger.error("Judge sample exhausted its retry budget: %s", sample_label)
            continue
        for name, value in sub_scores.items():
            totals[name] += value
        collected += 1

    if collected == 0:
        raise CandidateError(f"Judge produced no usable sample: {label}")

    means = {name: totals[name] / collected for name in rubric.names}
    return means, rubric.aggregate(means), collected


def _run_one_combo(
    persona_id: str,
    temp: float,
    rewriter_client: OpenAICompatClient,
    judge_client: OpenAICompatClient,
    rubric: Rubric,
    judge_samples: int,
    persona_rendered: str,
    question: QuestionContext,
    question_ref: str,
    rewrite_policy: RetryPolicy,
    judge_policy: RetryPolicy,
) -> Candidate | None:
    """Generate one rewrite and score it. Returns ``None`` when the candidate is unusable."""
    label = f"{question_ref} persona={persona_id} temp={temp:.1f}"
    rewriter = PromptedRewriter(rewriter_client)

    def _rewrite_once() -> str:
        rewrite = rewriter.rewrite(persona_rendered, question.query)
        if not rewrite.strip():
            raise ValueError("Rewriter returned an empty completion")
        return rewrite

    try:
        rewrite = _call_with_retry(f"Rewrite {label}", _rewrite_once, rewrite_policy)
    except CandidateError:
        logger.error("Dropping candidate — rewrite exhausted its retry budget: %s", label)
        return None

    messages = _build_judge_messages(
        rubric,
        persona_rendered,
        question.query,
        rewrite,
        answer=question.answer,
        explanation=question.explanation,
    )
    try:
        sub_scores, score, collected = _score_rewrite(
            judge_client, rubric, messages, judge_samples, judge_policy, f"Judge {label}"
        )
    except CandidateError:
        logger.error("Dropping candidate — judge produced no usable sample: %s", label)
        return None

    return Candidate(
        question_ref=question_ref,
        persona_id=persona_id,
        query=question.query,
        rewrite=rewrite,
        temperature=temp,
        sub_scores=sub_scores,
        score=score,
        judge_samples=collected,
    )


def _count_question_files(questions_dir: Path) -> int:
    count = sum(1 for _ in questions_dir.glob("*.json"))
    logger.info("Found %d question files in %s", count, questions_dir)
    return count


def _similarity(left: str, right: str) -> float:
    """Normalized similarity used to reject pairs that differ only cosmetically."""
    return difflib.SequenceMatcher(
        None, normalize_rewrite(left), normalize_rewrite(right), autojunk=False
    ).ratio()


def _apply_candidate_filters(candidates: list[Candidate], filters: Filters) -> list[Candidate]:
    """Mark and drop unusable candidates, mutating ``filter_reason`` on the rejects."""
    kept: list[Candidate] = []
    best_by_text: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if candidate.judge_samples < filters.min_judge_samples:
            candidate.filter_reason = "min_judge_samples"
            continue
        if filters.dedup_candidates:
            key = normalize_rewrite(candidate.rewrite)
            if key in best_by_text:
                candidate.filter_reason = "duplicate_rewrite"
                continue
            best_by_text[key] = candidate
        kept.append(candidate)
    return kept


def _serialize_pair(
    *,
    question_ref: str,
    query: str,
    persona_id: str,
    chosen: Candidate,
    rejected: Candidate,
    rejected_score: float,
    rejected_sub_scores: dict[str, float],
    pair_type: str,
) -> str:
    """Serialize one preference pair with the scores that produced it.

    ``rejected_score`` is passed in rather than read off ``rejected`` because a
    cross-persona negative is scored under the *target* persona, not under the persona it
    was written for; the row must record the comparison that was actually made.
    """
    record = {
        "format_version": DPO_OUTPUT_FORMAT_VERSION,
        "question_ref": question_ref,
        "persona_id": persona_id,
        "query": query,
        "chosen": chosen.rewrite,
        "rejected": rejected.rewrite,
        "pair_type": pair_type,
        "chosen_score": chosen.score,
        "rejected_score": rejected_score,
        "margin": chosen.score - rejected_score,
        "chosen_sub_scores": chosen.sub_scores,
        "rejected_sub_scores": rejected_sub_scores,
        "chosen_temperature": chosen.temperature,
        "rejected_temperature": rejected.temperature,
        "rejected_persona_id": rejected.persona_id,
        "judge_samples": min(chosen.judge_samples, rejected.judge_samples),
    }
    return json.dumps(record, ensure_ascii=False) + "\n"


def _select_within_persona_pair(ranked: Sequence[Candidate], filters: Filters) -> Candidate | None:
    """Pick the negative for the top candidate: worst first, skipping cosmetic near-twins.

    Scanning upward from the worst candidate keeps the largest real margin available
    instead of discarding the whole persona when the extreme pair happens to be a
    near-duplicate.
    """
    chosen = ranked[0]
    for candidate in reversed(ranked[1:]):
        if chosen.score - candidate.score < filters.min_margin:
            break
        if _similarity(chosen.rewrite, candidate.rewrite) >= filters.max_pair_similarity:
            continue
        return candidate
    return None


def _read_split_qids(splits_dir: Path, split_name: str) -> list[tuple[str, str]]:
    split_file = splits_dir / f"{split_name}_qids.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    pairs: list[tuple[str, str]] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            exam_stem, qid = line.split(":", 1)
        except ValueError:
            logger.warning("Invalid line in %s: %r", split_file, line)
            continue
        pairs.append((exam_stem.strip(), qid.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO rewriter training data")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--candidates-out",
        type=Path,
        default=None,
        help="Directory for the full candidate dump (default: data.candidates_dir)",
    )
    parser.add_argument(
        "--filtered-candidates-out",
        type=Path,
        default=None,
        help=(
            "Directory for candidates surviving the config filters "
            "(default: data.filtered_candidates_dir)"
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    random.seed(config["seed"])

    rewriter_cfg = config["rewriter"]
    judge_cfg = config["judge"]
    data_cfg = config["data"]
    filters = Filters.from_config(config["filters"])
    rubric = Rubric.from_config(judge_cfg["rubric"])

    judge_samples = judge_cfg["samples_per_candidate"]
    if isinstance(judge_samples, bool) or not isinstance(judge_samples, int) or judge_samples < 1:
        raise ValueError("judge.samples_per_candidate must be an integer >= 1")
    if filters.min_judge_samples > judge_samples:
        raise ValueError(
            f"filters.min_judge_samples ({filters.min_judge_samples}) exceeds "
            f"judge.samples_per_candidate ({judge_samples}): every candidate would be discarded"
        )

    temperatures: list[float] = rewriter_cfg.get("temperatures", [0.2, 0.5, 0.9])
    rewrite_policy = RetryPolicy.from_config(rewriter_cfg, "rewriter")
    judge_policy = RetryPolicy.from_config(judge_cfg, "judge")

    questions_dir = Path(data_cfg["questions_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    output_dir = Path(data_cfg["output_dir"])
    candidates_dir = args.candidates_out or Path(data_cfg["candidates_dir"])
    filtered_dir = args.filtered_candidates_out or Path(data_cfg["filtered_candidates_dir"])
    for directory in (output_dir, candidates_dir, filtered_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not _count_question_files(questions_dir):
        logger.error("No question files found in %s", questions_dir)
        return

    persona_ids = [p.id for p in train_personas()]
    persona_rendered = {pid: render_profile(pid) for pid in persona_ids}

    max_workers: int = rewriter_cfg.get("max_workers", 4)

    logger.info(
        "Initialising remote rewriter (%s) at %s", rewriter_cfg["model"], REWRITER_BASE_URL
    )
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

    judge_client = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=judge_cfg["model"],
        temperature=judge_cfg["temperature"],
        max_tokens=judge_cfg["max_completion_tokens"],
        reasoning_effort=judge_cfg.get("reasoning_effort"),
        response_format=rubric.response_format(),
    )

    logger.info(
        "Rewriter: %s temps=%s max_completion_tokens=%d max_workers=%d attempts=%d",
        rewriter_cfg["model"],
        temperatures,
        rewriter_cfg["max_completion_tokens"],
        max_workers,
        rewrite_policy.max_attempts,
    )
    logger.info(
        "Judge: %s reasoning_effort=%s temperature=%s max_completion_tokens=%d "
        "samples=%d attempts=%d rubric=%s",
        judge_cfg["model"],
        judge_cfg.get("reasoning_effort"),
        judge_cfg["temperature"],
        judge_cfg["max_completion_tokens"],
        judge_samples,
        judge_policy.max_attempts,
        ", ".join(f"{c.name}:{c.weight:g}" for c in rubric.criteria),
    )
    logger.info("Filters: %s", filters)

    for split_name in ("train", "val"):
        try:
            qid_pairs = _read_split_qids(splits_dir, split_name)
        except FileNotFoundError as e:
            logger.warning("%s — skipping %s split", e, split_name)
            continue

        if not qid_pairs:
            logger.warning("No question IDs found for split %r", split_name)
            continue

        output_path = output_dir / f"{split_name}.jsonl"
        candidates_path = candidates_dir / f"{split_name}.jsonl"
        filtered_path = filtered_dir / f"{split_name}.jsonl"
        records_written = 0
        candidates_written = 0
        filtered_written = 0
        rejected_by_filter: dict[str, int] = defaultdict(int)

        with (
            output_path.open("w", encoding="utf-8") as fh,
            candidates_path.open("w", encoding="utf-8") as candidates_fh,
            filtered_path.open("w", encoding="utf-8") as filtered_fh,
        ):
            for exam_stem, qid in tqdm(qid_pairs, desc=split_name, unit="q"):
                try:
                    question = load_question(exam_stem, qid, questions_dir)
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    logger.warning("Skipping %s:%s: %s", exam_stem, qid, e)
                    continue

                question_ref = f"{exam_stem}:{qid}"

                logger.info(
                    "Processing %s — %d chars — %s", question_ref, len(question.query), split_name
                )

                combos = [(pid, temp) for pid in persona_ids for temp in temperatures]
                all_candidates: list[Candidate] = []

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            _run_one_combo,
                            pid,
                            temp,
                            rewriter_clients[temp],
                            judge_client,
                            rubric,
                            judge_samples,
                            persona_rendered[pid],
                            question,
                            question_ref,
                            rewrite_policy,
                            judge_policy,
                        )
                        for pid, temp in combos
                    ]
                    for future in as_completed(futures):
                        candidate = future.result()
                        if candidate is not None:
                            all_candidates.append(candidate)

                by_persona: dict[str, list[Candidate]] = defaultdict(list)
                for candidate in all_candidates:
                    by_persona[candidate.persona_id].append(candidate)

                ranked_by_persona: dict[str, list[Candidate]] = {}
                for persona_id in persona_ids:
                    kept = _apply_candidate_filters(by_persona[persona_id], filters)
                    kept.sort(key=lambda c: c.score, reverse=True)
                    ranked_by_persona[persona_id] = kept

                best_by_persona = {
                    pid: ranked[0] for pid, ranked in ranked_by_persona.items() if ranked
                }

                # Cross-persona negatives are only meaningful when both sides are scored
                # under the same rubric context, so re-score each persona's winner as if it
                # had been written for every other persona. Without this the comparison
                # would put two scores from different persona prompts on one axis.
                for source_id, candidate in best_by_persona.items():
                    for target_id in persona_ids:
                        if target_id == source_id:
                            continue
                        label = f"Judge {question_ref} rewrite={source_id} as={target_id}"
                        messages = _build_judge_messages(
                            rubric,
                            persona_rendered[target_id],
                            question.query,
                            candidate.rewrite,
                            answer=question.answer,
                            explanation=question.explanation,
                        )
                        try:
                            sub_scores, score, collected = _score_rewrite(
                                judge_client,
                                rubric,
                                messages,
                                judge_samples,
                                judge_policy,
                                label,
                            )
                        except CandidateError:
                            logger.error("Cross-persona scoring failed: %s", label)
                            continue
                        candidate.scored_as[target_id] = {
                            "sub_scores": sub_scores,
                            "score": score,
                            "judge_samples": collected,
                        }

                for persona_id in persona_ids:
                    ranked = ranked_by_persona[persona_id]
                    if len(ranked) < 2:
                        logger.error(
                            "Only %d usable candidate(s) for %s persona=%s — no within-persona pair",
                            len(ranked),
                            question_ref,
                            persona_id,
                        )
                        rejected_by_filter["too_few_candidates"] += 1
                        continue

                    chosen = ranked[0]
                    if chosen.score < filters.min_chosen_score:
                        rejected_by_filter["min_chosen_score"] += 1
                        continue

                    rejected = _select_within_persona_pair(ranked, filters)
                    if rejected is None:
                        rejected_by_filter["min_margin_or_similarity"] += 1
                        continue

                    fh.write(
                        _serialize_pair(
                            question_ref=question_ref,
                            query=question.query,
                            persona_id=persona_id,
                            chosen=chosen,
                            rejected=rejected,
                            rejected_score=rejected.score,
                            rejected_sub_scores=rejected.sub_scores,
                            pair_type=WITHIN_PERSONA,
                        )
                    )
                    records_written += 1

                for target_id, target_best in best_by_persona.items():
                    if target_best.score < filters.min_chosen_score:
                        continue
                    for source_id, source_best in best_by_persona.items():
                        if source_id == target_id:
                            continue
                        foreign = source_best.scored_as.get(target_id)
                        if foreign is None:
                            continue
                        if target_best.score - foreign["score"] < filters.cross_persona_min_margin:
                            rejected_by_filter["cross_persona_min_margin"] += 1
                            continue
                        if (
                            _similarity(target_best.rewrite, source_best.rewrite)
                            >= filters.max_pair_similarity
                        ):
                            rejected_by_filter["cross_persona_similarity"] += 1
                            continue

                        fh.write(
                            _serialize_pair(
                                question_ref=question_ref,
                                query=question.query,
                                persona_id=target_id,
                                chosen=target_best,
                                rejected=source_best,
                                rejected_score=foreign["score"],
                                rejected_sub_scores=foreign["sub_scores"],
                                pair_type=CROSS_PERSONA,
                            )
                        )
                        records_written += 1

                kept_ids = {id(c) for ranked in ranked_by_persona.values() for c in ranked}
                for candidate in all_candidates:
                    candidates_fh.write(candidate.to_json())
                    candidates_written += 1
                    if id(candidate) in kept_ids:
                        filtered_fh.write(candidate.to_json())
                        filtered_written += 1
                    else:
                        rejected_by_filter[candidate.filter_reason or "unknown"] += 1

        logger.info("Wrote %d pair records to %s", records_written, output_path)
        logger.info("Wrote %d candidates to %s", candidates_written, candidates_path)
        logger.info("Wrote %d filtered candidates to %s", filtered_written, filtered_path)
        if rejected_by_filter:
            logger.info(
                "Filter rejections (%s): %s",
                split_name,
                ", ".join(f"{k}={v}" for k, v in sorted(rejected_by_filter.items())),
            )


if __name__ == "__main__":
    main()
