"""Generate DPO rewriter preference pairs: N rewrites per (query, persona) ranked by LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml
from tqdm import tqdm

from data.questions import QuestionContext, load_question, render_question_value
from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITER_API_KEY, REWRITER_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.llm import OpenAICompatClient
from rag.rewriter import PromptedRewriter

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are an expert Persian language tutor evaluating query rewrites for a RAG retrieval system."

# Bumped whenever the meaning of a written row changes. Version 1 is the first format
# whose ``query`` is the complete rendered question (data.questions.load_question) rather
# than the bare stem, and whose scores come from a judge that saw the gold answer.
# ``rl.dpo_train.load_pairs`` refuses any other version, so stem-only pairs cannot be
# trained on by accident.
DPO_OUTPUT_FORMAT_VERSION = 1

# A judge reply must be exactly one score in [0, 1] — nothing looser, so that a truncated
# or chatty response retries instead of silently becoming a real-looking score.
SCORE_RE = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?)")

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
        "Rate 0.0–1.0 how well this rewrite would help retrieve the right study material "
        "for this specific student. A good rewrite should:\n"
        "  - Preserve the original question's meaning\n"
        "  - Use vocabulary and framing that matches the student's profile\n"
        "  - Be specific enough to surface relevant passages at the right depth\n\n"
        "Respond with a single decimal number only, e.g. 0.61"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse_score(response: str) -> float:
    """Parse a judge response containing exactly one score in ``[0, 1]``."""
    if not isinstance(response, str):
        raise ValueError(f"Judge response is not a score string: {response!r}")
    stripped = response.strip()
    if SCORE_RE.fullmatch(stripped) is None:
        raise ValueError(f"Judge response is not a single score in [0, 1]: {response!r}")
    score = float(stripped)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge response is not a finite score in [0, 1]: {response!r}")
    return score


def _run_one_combo(
    persona_id: str,
    temp: float,
    rewriter_client: OpenAICompatClient,
    judge_client: OpenAICompatClient,
    persona_rendered: str,
    question: QuestionContext,
    exam_stem: str,
    qid: str,
    rewrite_policy: RetryPolicy,
    judge_policy: RetryPolicy,
) -> tuple[str, str | None, float]:
    """Generate one rewrite and score it. Returns (persona_id, rewrite_or_None, score)."""
    label = f"{exam_stem}:{qid} persona={persona_id} temp={temp:.1f}"
    rewriter = PromptedRewriter(rewriter_client)

    def _rewrite_once() -> str:
        rewrite = rewriter.rewrite(persona_rendered, question.query)
        if not rewrite.strip():
            raise ValueError("Rewriter returned an empty completion")
        return rewrite

    def _score_once() -> float:
        messages = _build_judge_messages(
            persona_rendered,
            question.query,
            rewrite,
            answer=question.answer,
            explanation=question.explanation,
        )
        return _parse_score(judge_client.chat(messages))

    try:
        rewrite = _call_with_retry(f"Rewrite {label}", _rewrite_once, rewrite_policy)
    except CandidateError:
        logger.error("Dropping candidate — rewrite exhausted its retry budget: %s", label)
        return persona_id, None, 0.0

    try:
        score = _call_with_retry(f"Judge {label}", _score_once, judge_policy)
    except CandidateError:
        logger.error("Dropping candidate — judge exhausted its retry budget: %s", label)
        return persona_id, None, 0.0

    return persona_id, rewrite, score


def _count_question_files(questions_dir: Path) -> int:
    count = sum(1 for _ in questions_dir.glob("*.json"))
    logger.info("Found %d question files in %s", count, questions_dir)
    return count


def _serialize_record(
    question_ref: str, query: str, persona_id: str, chosen: str, rejected: str
) -> str:
    """Serialize one preference pair, stamped so its provenance is readable from the file."""
    record = {
        "format_version": DPO_OUTPUT_FORMAT_VERSION,
        "question_ref": question_ref,
        "persona_id": persona_id,
        "query": query,
        "chosen": chosen,
        "rejected": rejected,
    }
    return json.dumps(record, ensure_ascii=False) + "\n"


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
    cross_threshold = config["cross_persona_threshold"]

    temperatures: list[float] = rewriter_cfg.get("temperatures", [0.2, 0.5, 0.9])
    rewrite_policy = RetryPolicy.from_config(rewriter_cfg, "rewriter")
    judge_policy = RetryPolicy.from_config(judge_cfg, "judge")

    questions_dir = Path(data_cfg["questions_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _count_question_files(questions_dir):
        logger.error("No question files found in %s", questions_dir)
        return

    persona_ids = [p.id for p in train_personas()]

    max_workers: int = rewriter_cfg.get("max_workers", 4)

    logger.info("Initialising remote rewriter (%s) at %s", rewriter_cfg["model"], REWRITER_BASE_URL)
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
        "Judge: %s reasoning_effort=%s temperature=%s max_completion_tokens=%d attempts=%d",
        judge_cfg["model"],
        judge_cfg.get("reasoning_effort"),
        judge_cfg["temperature"],
        judge_cfg["max_completion_tokens"],
        judge_policy.max_attempts,
    )

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
        records_written = 0

        with output_path.open("w", encoding="utf-8") as fh:
            for exam_stem, qid in tqdm(qid_pairs, desc=split_name, unit="q"):
                try:
                    question = load_question(exam_stem, qid, questions_dir)
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    logger.warning("Skipping %s:%s: %s", exam_stem, qid, e)
                    continue

                question_ref = f"{exam_stem}:{qid}"

                logger.info(
                    "Processing %s:%s — %d chars — %s",
                    exam_stem,
                    qid,
                    len(question.query),
                    split_name,
                )

                combos = [(pid, temp) for pid in persona_ids for temp in temperatures]
                candidates_by_persona: dict[str, list[tuple[str, float]]] = defaultdict(list)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_combo = {
                        executor.submit(
                            _run_one_combo,
                            pid,
                            temp,
                            rewriter_clients[temp],
                            judge_client,
                            render_profile(pid),
                            question,
                            exam_stem,
                            qid,
                            rewrite_policy,
                            judge_policy,
                        ): (pid, temp)
                        for pid, temp in combos
                    }
                    for future in as_completed(future_to_combo):
                        pid, rewrite, score = future.result()
                        if rewrite is not None:
                            candidates_by_persona[pid].append((rewrite, score))

                best_for_persona: dict[str, str] = {}
                best_score_for_persona: dict[str, float] = {}

                for persona_id in persona_ids:
                    candidates = candidates_by_persona[persona_id]
                    if len(candidates) < 2:
                        logger.error(
                            "Only %d candidate(s) for %s:%s persona=%s — skipping persona",
                            len(candidates),
                            exam_stem,
                            qid,
                            persona_id,
                        )
                        continue

                    candidates.sort(key=lambda x: x[1], reverse=True)
                    chosen, chosen_score = candidates[0]
                    rejected, _ = candidates[-1]

                    best_for_persona[persona_id] = chosen
                    best_score_for_persona[persona_id] = chosen_score

                    fh.write(
                        _serialize_record(
                            question_ref, question.query, persona_id, chosen, rejected
                        )
                    )
                    records_written += 1

                for i, pid_a in enumerate(persona_ids):
                    if pid_a not in best_for_persona:
                        continue
                    for pid_b in persona_ids[i + 1 :]:
                        if pid_b not in best_for_persona:
                            continue
                        if (
                            abs(best_score_for_persona[pid_a] - best_score_for_persona[pid_b])
                            <= cross_threshold
                        ):
                            continue

                        fh.write(
                            _serialize_record(
                                question_ref,
                                question.query,
                                pid_a,
                                best_for_persona[pid_a],
                                best_for_persona[pid_b],
                            )
                        )
                        records_written += 1

                        fh.write(
                            _serialize_record(
                                question_ref,
                                question.query,
                                pid_b,
                                best_for_persona[pid_b],
                                best_for_persona[pid_a],
                            )
                        )
                        records_written += 1

        logger.info("Wrote %d records to %s", records_written, output_path)


if __name__ == "__main__":
    main()
