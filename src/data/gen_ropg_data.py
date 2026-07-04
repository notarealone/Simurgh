"""Generate ROPG-KD training data (query, persona) → top-K chunks scored by LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import yaml

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


def _score_chunks(
    judge: OpenAICompatClient,
    query: str,
    persona_rendered: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> list[dict]:
    """Score each chunk with the LLM judge. Returns docs ordered by input rank."""
    docs: list[dict] = []
    for cid, ctext in zip(chunk_ids, chunk_texts, strict=True):
        messages = _build_judge_messages(query, persona_rendered, ctext)
        response = judge.chat(messages)
        score = _parse_float_score(response)
        docs.append({"chunk_id": cid, "text": ctext, "teacher_score": score})
    return docs


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
    sims = query_vec @ chunk_matrix.T
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
    )
    chunk_matrix = embedder.encode(chunk_texts)

    logger.info("Initialising LLM judge (%s)", cfg["judge"]["model"])
    judge = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=cfg["judge"]["model"],
        temperature=cfg["judge"]["temperature"],
        max_tokens=cfg["judge"]["max_tokens"],
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
        with output_path.open("w", encoding="utf-8") as out_fh:
            for exam_stem, qid, raw_line in entries:
                try:
                    query = _load_question(exam_stem, qid, questions_dir)
                except (FileNotFoundError, KeyError) as exc:
                    logger.warning("Skipping %s: %s", raw_line, exc)
                    continue

                for persona in train_profiles:
                    persona_rendered = render_profile(persona.id)
                    query_vec = embedder.encode_query(
                        texts=[query],
                        instruction=persona_rendered,
                    )
                    top_ids, top_texts = _retrieve_top_k(
                        query_vec, chunk_matrix, top_k, chunk_ids, chunk_texts
                    )
                    docs = _score_chunks(judge, query, persona_rendered, top_ids, top_texts)
                    rec = {
                        "query": query,
                        "persona_id": persona.id,
                        "docs": docs,
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                logger.info("Processed %s (%s)", raw_line, split)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate ROPG-KD training data")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    run(args.config)


if __name__ == "__main__":
    main()
