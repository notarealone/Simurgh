"""Generate ROPG-KD training data (query, persona) → top-K chunks scored by LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from data.questions import QuestionContext, load_question, render_question_value
from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.embedder import Qwen3Embedder
from rag.llm import OpenAICompatClient

logger = logging.getLogger(__name__)

SCORE_RE = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?)")
QUESTION_OUTPUT_FORMAT_VERSION = 4


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Return (chunk_ids, texts) loaded from a JSONL corpus file."""
    chunk_ids: list[str] = []
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        chunk_ids.append(rec["chunk_id"])
        texts.append(rec["text"])
    if not texts:
        raise ValueError(f"Corpus file {path} is empty or has no valid records")
    return chunk_ids, texts


def _parse_float_score(response: str) -> float:
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


def _build_judge_messages(
    query: str,
    persona_rendered: str,
    chunk_text: str,
    answer: object | None = None,
    explanation: str | None = None,
) -> list[dict[str, str]]:
    # Gold fields are passed separately to the judge and never become part of query.
    reference_sections: list[str] = []
    if answer is not None:
        rendered_answer = answer if isinstance(answer, str) else render_question_value(answer)
        reference_sections.append(f"Gold answer/reference:\n{rendered_answer}")
    if explanation is not None:
        reference_sections.append(f"Gold explanation/rubric:\n{explanation}")
    reference_context = ""
    if reference_sections:
        reference_context = (
            "Reference answer and rubric (judge context only; not part of the retrieval query):\n"
            + "\n\n".join(reference_sections)
            + "\n\n"
        )

    system = "You are an expert Persian language tutor evaluating study materials."
    user = (
        "A student with the following profile is trying to answer an exam question:\n"
        f"Profile: {persona_rendered}\n\n"
        f"Exam question: {query}\n\n"
        f"{reference_context}"
        f"Candidate study passage:\n{chunk_text}\n\n"
        "Rate 0.0–1.0 how useful this passage is for helping this specific student answer "
        "the question. Consider:\n"
        "  - Does the depth match the student's comprehension level?\n"
        "  - Does it provide what this student needs (simple paraphrase vs. deep analysis)?\n"
        "  - Is the style appropriate (hand-holding vs. terse treatment)?\n\n"
        "Respond with a single decimal number only, e.g. 0.73"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class JudgeScoringError(RuntimeError):
    """Raised when a chunk cannot be scored within the configured retry budget."""


def _score_one_chunk(
    judge: OpenAICompatClient,
    query: str,
    persona_rendered: str,
    cid: str,
    ctext: str,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff_seconds: float = 8.0,
    answer: object | None = None,
    explanation: str | None = None,
) -> dict:
    messages = _build_judge_messages(
        query,
        persona_rendered,
        ctext,
        answer=answer,
        explanation=explanation,
    )
    backoff_seconds = initial_backoff_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            response = judge.chat(messages)
            score = _parse_float_score(response)
            return {"chunk_id": cid, "text": ctext, "teacher_score": score}
        except Exception as exc:
            logger.warning(
                "Judge attempt failed for chunk %s (attempt %d/%d)",
                cid,
                attempt,
                max_attempts,
                exc_info=True,
            )
            if attempt == max_attempts:
                raise JudgeScoringError(
                    f"Failed to score chunk {cid!r} after {max_attempts} attempts"
                ) from exc
            time.sleep(backoff_seconds)
            backoff_seconds = min(
                backoff_seconds * backoff_multiplier,
                max_backoff_seconds,
            )

    raise JudgeScoringError(f"Failed to score chunk {cid!r}")


def _score_chunks(
    judge: OpenAICompatClient,
    query: str,
    persona_rendered: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    max_workers: int = 8,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff_seconds: float = 8.0,
    answer: object | None = None,
    explanation: str | None = None,
) -> list[dict]:
    """Score chunks in parallel. Returns docs in the same order as input."""
    results: list[dict | None] = [None] * len(chunk_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _score_one_chunk,
                judge,
                query,
                persona_rendered,
                cid,
                ctext,
                max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds,
                backoff_multiplier=backoff_multiplier,
                max_backoff_seconds=max_backoff_seconds,
                answer=answer,
                explanation=explanation,
            ): index
            for index, (cid, ctext) in enumerate(zip(chunk_ids, chunk_texts, strict=True))
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            doc = future.result()
            results[index] = doc
    if any(doc is None for doc in results):
        raise RuntimeError("Chunk scoring completed without a result for every input chunk")
    return [doc for doc in results if doc is not None]


def _load_split_qids(split_path: Path) -> list[tuple[str, str, str]]:
    """Parse ``{exam_stem}:{qid}`` lines into (exam_stem, qid, line)."""
    entries: list[tuple[str, str, str]] = []
    for raw in split_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            logger.warning("Skipping malformed split line: %r", line)
            continue
        exam_stem, qid = line.split(":", 1)
        entries.append((exam_stem, qid, line))
    return entries


def _retrieve_top_k(
    query_vec: np.ndarray,
    chunk_matrix: np.ndarray,
    top_k: int,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> tuple[list[str], list[str]]:
    sims = query_vec.squeeze() @ chunk_matrix.T
    top_idx = np.argsort(sims)[-top_k:][::-1]
    ids = [chunk_ids[i] for i in top_idx]
    texts = [chunk_texts[i] for i in top_idx]
    return ids, texts


def run(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    judge_cfg = cfg["judge"]
    max_attempts = judge_cfg.get("max_attempts", 3)
    initial_backoff_seconds = judge_cfg.get("initial_backoff_seconds", 1.0)
    backoff_multiplier = judge_cfg.get("backoff_multiplier", 2.0)
    max_backoff_seconds = judge_cfg.get("max_backoff_seconds", 8.0)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("judge.max_attempts must be an integer >= 1")
    for setting_name, setting_value in (
        ("initial_backoff_seconds", initial_backoff_seconds),
        ("backoff_multiplier", backoff_multiplier),
        ("max_backoff_seconds", max_backoff_seconds),
    ):
        if isinstance(setting_value, bool) or not isinstance(setting_value, (int, float)):
            raise ValueError(f"judge.{setting_name} must be a finite number")
        if not math.isfinite(float(setting_value)):
            raise ValueError(f"judge.{setting_name} must be a finite number")
    if initial_backoff_seconds < 0:
        raise ValueError("judge.initial_backoff_seconds must be >= 0")
    if backoff_multiplier < 1:
        raise ValueError("judge.backoff_multiplier must be >= 1")
    if max_backoff_seconds < initial_backoff_seconds:
        raise ValueError(
            "judge.max_backoff_seconds must be >= judge.initial_backoff_seconds"
        )

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    corpus_path = Path(cfg["data"]["chunks"])
    questions_dir = Path(cfg["data"]["questions_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["retriever"]["top_k"]
    max_workers = judge_cfg.get("max_workers", 8)

    logger.info("Loading corpus from %s", corpus_path)
    chunk_ids, chunk_texts = _load_corpus(corpus_path)

    logger.info(
        "Encoding corpus (%d chunks) with %s on %s",
        len(chunk_texts),
        cfg["embedder"]["model"],
        cfg["embedder"]["device"],
    )
    embedder = Qwen3Embedder(
        model_name=cfg["embedder"]["model"],
        device=cfg["embedder"]["device"],
        batch_size=cfg["embedder"].get("batch_size", 32),
        fp16=cfg["embedder"].get("fp16", False),
        max_seq_length=cfg["embedder"]["max_seq_length"],
    )
    chunk_matrix = embedder.encode(chunk_texts)

    logger.info("Initialising LLM judge (%s)", judge_cfg["model"])
    judge = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=judge_cfg["model"],
        temperature=judge_cfg["temperature"],
        max_tokens=judge_cfg["max_completion_tokens"],
        reasoning_effort=judge_cfg.get("reasoning_effort"),
    )

    train_profiles = train_personas()

    for split in ("train", "val"):
        split_path = splits_dir / f"{split}_qids.txt"
        if not split_path.exists():
            logger.warning("Split file not found: %s, skipping", split_path)
            continue

        output_path = output_dir / f"{split}.jsonl"
        logger.info("Processing %s split → %s", split, output_path)

        entries = _load_split_qids(split_path)

        # Build resume set from existing output
        seen: set[tuple[str, str]] = set()
        if output_path.exists():
            for line_number, raw in enumerate(
                output_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in existing output {output_path} at line "
                        f"{line_number}; refusing to resume"
                    ) from exc
                if not isinstance(rec, dict):
                    raise ValueError(
                        f"Existing output {output_path} line {line_number} is not a JSON "
                        "object; refusing to resume"
                    )
                if "format_version" not in rec:
                    raise ValueError(
                        f"Existing output {output_path} line {line_number} has no format "
                        "version; refusing to append"
                    )
                if rec["format_version"] != QUESTION_OUTPUT_FORMAT_VERSION:
                    raise ValueError(
                        f"Existing output {output_path} line {line_number} has format "
                        f"version {rec['format_version']!r}, expected "
                        f"{QUESTION_OUTPUT_FORMAT_VERSION}; refusing to append"
                    )
                try:
                    seen.add((rec["query"], rec["persona_id"]))
                except KeyError as exc:
                    raise ValueError(
                        f"Existing output {output_path} line {line_number} is missing "
                        "query or persona_id; refusing to resume"
                    ) from exc
            if seen:
                logger.info("Resuming %s: %d records already written", split, len(seen))

        failed_groups = 0
        with output_path.open("a", encoding="utf-8") as out_fh:
            # Pre-load all valid complete question contexts for this split.
            valid_entries: list[tuple[str, str, str, QuestionContext]] = []
            for exam_stem, qid, raw_line in entries:
                try:
                    question = load_question(exam_stem, qid, questions_dir)
                    valid_entries.append((exam_stem, qid, raw_line, question))
                except (FileNotFoundError, KeyError) as exc:
                    logger.warning("Skipping %s: %s", raw_line, exc)

            for persona in train_profiles:
                persona_rendered = render_profile(persona.id)
                todo = [
                    (i, e)
                    for i, e in enumerate(valid_entries)
                    if (e[3].query, persona.id) not in seen
                ]
                if not todo:
                    continue
                query_vecs = embedder.encode_query(
                    [e[3].query for _, e in todo],
                    instruction=persona_rendered,
                )
                for j, (_, (_, _, _raw_line, question)) in enumerate(
                    tqdm(todo, desc=f"{split}/{persona.id}", unit="q")
                ):
                    query_vec = query_vecs[j : j + 1]
                    top_ids, top_texts = _retrieve_top_k(
                        query_vec, chunk_matrix, top_k, chunk_ids, chunk_texts
                    )
                    try:
                        docs = _score_chunks(
                            judge,
                            question.query,
                            persona_rendered,
                            top_ids,
                            top_texts,
                            max_workers=max_workers,
                            max_attempts=max_attempts,
                            initial_backoff_seconds=initial_backoff_seconds,
                            backoff_multiplier=backoff_multiplier,
                            max_backoff_seconds=max_backoff_seconds,
                            answer=question.answer,
                            explanation=question.explanation,
                        )
                    except JudgeScoringError as exc:
                        failed_groups += 1
                        logger.error(
                            "Skipping failed scoring group split=%s question=%s persona=%s: %s",
                            split,
                            _raw_line,
                            persona.id,
                            exc,
                        )
                        continue
                    rec = {
                        "format_version": QUESTION_OUTPUT_FORMAT_VERSION,
                        "question_ref": _raw_line,
                        "query": question.query,
                        "persona_id": persona.id,
                        "docs": docs,
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()

        logger.info("Finished %s split: %d failed scoring groups", split, failed_groups)

        # Convert scored data to hard-negative triplets/pairs for hard_neg training mode.
        # Read from the `triplets:` block — this used to read cfg["max_negatives"] at the
        # top level, where no such key has ever existed, so the knob was pinned at 4
        # regardless of config.
        triplet_cfg = cfg.get("triplets", {})
        derive_triplets(
            output_path,
            output_dir,
            max_negatives=triplet_cfg.get("max_negatives", 4),
            filters=triplet_cfg.get("filters"),
        )


# The scored data is the ground truth: each (query, persona) group has docs ranked
# by an LLM judge (teacher_score in [0,1]). Triplets are derived by treating the
# highest-scored doc as the positive and the lowest-scored docs as negatives —
# a coarse binarisation of the continuous signal that lets the same dataset drive
# MNRL (hard_neg mode) training without a separate annotation pass.
#
# That binarisation is *why* hard_neg survives this teacher and listwise KD does not.
# Measured on train.jsonl, mean teacher_score by rank runs 0.76 0.60 0.49 ... 0.04 0.03:
# the rank-1 vs rank-17..20 gap is 0.711 (sd 0.183), far above the judge's noise floor,
# while adjacent gaps in the middle of the list are 0.02-0.03 — below it. KD's softmax
# weights the whole curve and so spends most of its gradient reproducing coin flips;
# MNRL only ever reads the extremes. See docs/methodology.md.
def derive_triplets(
    scored_path: Path,
    output_dir: Path,
    max_negatives: int = 4,
    filters: dict | None = None,
) -> None:
    """Derive hard-negative triplets and pairs from a scored JSONL file.

    Writes two files alongside the scored file:
      {stem}_triplets.jsonl — one line per group: {query, persona_id, positive, negatives:[...]}
      {stem}_pairs.jsonl    — one line per (pos, neg) pair: {query, persona_id, positive, negative}

    Negatives are written **hardest-first** (descending teacher_score), which matters
    because ``TripletDataset`` truncates with ``negatives[:max_negatives]``: a file
    holding more negatives than training requests must hand over the hardest ones,
    not the easiest.

    *filters* drops groups whose supervision is noise rather than signal — an arbitrary
    positive, or a group where the judge found nothing useful at all. It defaults to
    disabled, so the unfiltered arm stays reproducible from config alone.

    Format validation stays in this function so normal generation, ``--derive-only``, and
    the generated Kaggle notebook all reject stale scored data before replacing derived files.
    """
    stem = scored_path.stem  # e.g. "train" or "val"
    triplet_path = output_dir / f"{stem}_triplets.jsonl"
    pairs_path = output_dir / f"{stem}_pairs.jsonl"

    # Validate the complete input before opening derived files in truncate mode. In
    # particular, stem-only data predating complete question rendering must never be
    # made to look current by running --derive-only.
    scored_records: list[dict] = []
    for line_number, raw in enumerate(
        scored_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in scored data {scored_path} at line {line_number}"
            ) from exc
        if not isinstance(rec, dict):
            raise ValueError(
                f"Scored data {scored_path} line {line_number} is not a JSON object"
            )
        if rec.get("format_version") != QUESTION_OUTPUT_FORMAT_VERSION:
            raise ValueError(
                f"Scored data {scored_path} line {line_number} has format version "
                f"{rec.get('format_version')!r}, expected {QUESTION_OUTPUT_FORMAT_VERSION}; "
                "regenerate scored data instead of using --derive-only"
            )
        scored_records.append(rec)

    filters = filters or {}
    enabled = bool(filters.get("enabled", False))
    # Thresholds are read once here so the resolved values land in the sidecar even
    # when filtering is off — that is what makes a derived file traceable to a config.
    min_positive_margin = float(filters.get("min_positive_margin", 0.08))
    min_positive_score = float(filters.get("min_positive_score", 0.4))
    min_negative_margin = float(filters.get("min_negative_margin", 0.3))

    n_groups = 0
    n_triplets = 0
    n_pairs = 0
    n_dropped = {"too_few_docs": 0, "positive_margin": 0, "positive_score": 0, "no_negatives": 0}

    with (
        triplet_path.open("w", encoding="utf-8") as tf,
        pairs_path.open("w", encoding="utf-8") as pf,
    ):
        for rec in scored_records:
            docs = rec.get("docs", [])
            n_groups += 1
            if len(docs) < 2:
                n_dropped["too_few_docs"] += 1
                continue

            sorted_docs = sorted(docs, key=lambda d: d["teacher_score"], reverse=True)
            pos_score = sorted_docs[0]["teacher_score"]

            if enabled:
                # A positive that ties with rank-2 is arbitrary: the two scores differ by
                # less than the judge's noise, so which one becomes "the positive" is a coin
                # flip. Measured at 34% of train groups for a 0.05 margin.
                if pos_score - sorted_docs[1]["teacher_score"] < min_positive_margin:
                    n_dropped["positive_margin"] += 1
                    continue
                # If even the best candidate is weak, the judge found nothing useful in the
                # retrieved pool and the positive is noise by construction. 10% of train
                # groups score below 0.5 at rank 1.
                if pos_score < min_positive_score:
                    n_dropped["positive_score"] += 1
                    continue

            positive = sorted_docs[0]["text"]
            # Slice off rank-1 before taking the tail: with max_negatives >= len(docs) the
            # window would otherwise reach back far enough to include the positive itself
            # and emit it as its own negative. Real groups hold 20 docs so this does not
            # bite at max_negatives=8, but it is one config bump away from doing so.
            #
            # The tail stays in descending score order, which is already hardest-first:
            # index 0 is the highest-scoring doc the judge still ranked out of contention,
            # so TripletDataset's ``negatives[:max_negatives]`` head slice keeps the hardest
            # ones when a file holds more negatives than training asks for.
            tail = sorted_docs[1:][-max_negatives:]
            if enabled:
                tail = [d for d in tail if pos_score - d["teacher_score"] >= min_negative_margin]
            negatives = [d["text"] for d in tail]
            if not negatives:
                n_dropped["no_negatives"] += 1
                continue

            base = {"query": rec["query"], "persona_id": rec.get("persona_id", "")}

            tf.write(
                json.dumps(
                    {**base, "positive": positive, "negatives": negatives}, ensure_ascii=False
                )
                + "\n"
            )
            n_triplets += 1

            for neg in negatives:
                pf.write(
                    json.dumps({**base, "positive": positive, "negative": neg}, ensure_ascii=False)
                    + "\n"
                )
                n_pairs += 1

    logger.info(
        "Derived %d triplets and %d pairs from %s (%d/%d groups retained) → %s, %s",
        n_triplets,
        n_pairs,
        scored_path.name,
        n_triplets,
        n_groups,
        triplet_path.name,
        pairs_path.name,
    )
    if enabled:
        logger.info("  dropped: %s", ", ".join(f"{k}={v}" for k, v in n_dropped.items() if v))

    # Sidecar: a derived file is otherwise indistinguishable from one built under
    # different thresholds, and the retained counts go straight into the results table.
    meta_path = output_dir / f"{stem}_triplets_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source": scored_path.name,
                "max_negatives": max_negatives,
                "filters": {
                    "enabled": enabled,
                    "min_positive_margin": min_positive_margin,
                    "min_positive_score": min_positive_score,
                    "min_negative_margin": min_negative_margin,
                },
                "groups_total": n_groups,
                "groups_retained": n_triplets,
                "pairs": n_pairs,
                "dropped": n_dropped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def derive_only(config_path: str | Path) -> None:
    """Rebuild triplets/pairs from already-scored data, without judging anything.

    The scored ``{train,val}.jsonl`` are the expensive artefact — one judge call per
    (query, persona, chunk). Everything downstream of them (``triplets.max_negatives``,
    the label filters) is a pure re-read, so changing those settings should never require
    paying for the corpus to be re-judged. This entry point exists to make that explicit.
    """
    with Path(config_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    output_dir = Path(cfg["data"]["output_dir"])
    triplet_cfg = cfg.get("triplets", {})
    max_negatives = triplet_cfg.get("max_negatives", 4)
    filters = triplet_cfg.get("filters")

    logger.info(
        "Deriving from %s | max_negatives=%d | filters=%s",
        output_dir,
        max_negatives,
        (filters or {}).get("enabled", False),
    )

    found = False
    for split in ("train", "val"):
        scored_path = output_dir / f"{split}.jsonl"
        if not scored_path.exists():
            logger.warning("%s not found — skipping", scored_path)
            continue
        found = True
        derive_triplets(scored_path, output_dir, max_negatives=max_negatives, filters=filters)

    if not found:
        raise FileNotFoundError(
            f"No scored data in {output_dir}. Run without --derive-only first."
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate ROPG-KD training data")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--derive-only",
        action="store_true",
        help=(
            "Skip judging and only rebuild the triplet/pair files from the existing "
            "{train,val}.jsonl. Makes zero API calls and loads no model, so it is the "
            "cheap way to change triplets.max_negatives or the label filters."
        ),
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    if args.derive_only:
        derive_only(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
