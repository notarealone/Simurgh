"""Generate ROPG-KD training data (query, persona) → top-K chunks scored by LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.embedder import Qwen3Embedder
from rag.llm import OpenAICompatClient

logger = logging.getLogger(__name__)

FLOAT_RE = re.compile(r"[\d.]+")


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
    """Extract the first float from the LLM response and clamp to [0, 1]."""
    m = FLOAT_RE.search(response)
    if m is None:
        logger.warning("Could not parse float from judge response: %r", response)
        return 0.0
    try:
        score = float(m.group())
    except ValueError:
        logger.warning("Could not parse float from judge response: %r", response)
        return 0.0
    return max(0.0, min(1.0, score))


def _build_judge_messages(
    query: str, persona_rendered: str, chunk_text: str
) -> list[dict[str, str]]:
    system = "You are an expert Persian language tutor evaluating study materials."
    user = (
        "A student with the following profile is trying to answer an exam question:\n"
        f"Profile: {persona_rendered}\n\n"
        f"Exam question: {query}\n\n"
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


def _score_one_chunk(
    judge: OpenAICompatClient,
    query: str,
    persona_rendered: str,
    cid: str,
    ctext: str,
) -> dict:
    messages = _build_judge_messages(query, persona_rendered, ctext)
    try:
        response = judge.chat(messages)
        score = _parse_float_score(response)
    except Exception:
        logger.warning("Judge call failed for chunk %s", cid, exc_info=True)
        score = 0.0
    return {"chunk_id": cid, "text": ctext, "teacher_score": score}


def _score_chunks(
    judge: OpenAICompatClient,
    query: str,
    persona_rendered: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    max_workers: int = 8,
) -> list[dict]:
    """Score chunks in parallel. Returns docs in the same order as input."""
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cid = {
            executor.submit(_score_one_chunk, judge, query, persona_rendered, cid, ctext): cid
            for cid, ctext in zip(chunk_ids, chunk_texts, strict=True)
        }
        for future in as_completed(future_to_cid):
            doc = future.result()
            results[doc["chunk_id"]] = doc
    return [results[cid] for cid in chunk_ids]


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


def _load_question(exam_stem: str, qid: str, questions_dir: Path) -> str:
    """Load a question JSON and return the *stem* text for *qid*."""
    qfile = questions_dir / f"{exam_stem}.json"
    if not qfile.exists():
        raise FileNotFoundError(f"Question file not found: {qfile}")
    data = json.loads(qfile.read_text(encoding="utf-8"))
    for q in data.get("questions", []):
        if q["id"] == qid:
            return q["stem"]
    raise KeyError(f"Question {qid!r} not found in {qfile}")


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

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    corpus_path = Path(cfg["data"]["chunks"])
    questions_dir = Path(cfg["data"]["questions_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["retriever"]["top_k"]
    max_workers = cfg["judge"].get("max_workers", 8)

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
    )
    chunk_matrix = embedder.encode(chunk_texts)

    logger.info("Initialising LLM judge (%s)", cfg["judge"]["model"])
    judge = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=cfg["judge"]["model"],
        temperature=cfg["judge"]["temperature"],
        max_completion_tokens=cfg["judge"]["max_completion_tokens"],
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
            for raw in output_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                    seen.add((rec["query"], rec["persona_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
            if seen:
                logger.info("Resuming %s: %d records already written", split, len(seen))

        with output_path.open("a", encoding="utf-8") as out_fh:
            # Pre-load all valid question stems for this split
            valid_entries: list[tuple[str, str, str, str]] = []
            for exam_stem, qid, raw_line in entries:
                try:
                    query = _load_question(exam_stem, qid, questions_dir)
                    valid_entries.append((exam_stem, qid, raw_line, query))
                except (FileNotFoundError, KeyError) as exc:
                    logger.warning("Skipping %s: %s", raw_line, exc)

            for persona in train_profiles:
                persona_rendered = render_profile(persona.id)
                todo = [
                    (i, e) for i, e in enumerate(valid_entries) if (e[3], persona.id) not in seen
                ]
                if not todo:
                    continue
                query_vecs = embedder.encode_query(
                    [e[3] for _, e in todo],
                    instruction=persona_rendered,
                )
                for j, (_, (_, _, _raw_line, query)) in enumerate(
                    tqdm(todo, desc=f"{split}/{persona.id}", unit="q")
                ):
                    query_vec = query_vecs[j : j + 1]
                    top_ids, top_texts = _retrieve_top_k(
                        query_vec, chunk_matrix, top_k, chunk_ids, chunk_texts
                    )
                    docs = _score_chunks(
                        judge, query, persona_rendered, top_ids, top_texts, max_workers
                    )
                    rec = {
                        "query": query,
                        "persona_id": persona.id,
                        "docs": docs,
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()

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

    Everything this function needs is defined inside it. ``notebooks/build_train_ropg_kd.py``
    inlines it into the Kaggle worker by slicing from this function's header to the next
    top-level definition, so a module-level helper would be silently dropped there and fail
    with a NameError only on the notebook's lazy re-derive path.
    """
    stem = scored_path.stem  # e.g. "train" or "val"
    triplet_path = output_dir / f"{stem}_triplets.jsonl"
    pairs_path = output_dir / f"{stem}_pairs.jsonl"

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
        for raw in scored_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
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
