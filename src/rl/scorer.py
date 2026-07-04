"""Offline LLM-judge scoring of (query, persona, document) triples for ROPG-KD.

For each (question, persona) in the train split, retrieves the top-K candidate
documents from the dense index and asks an LLM judge to score each document's
pedagogical utility for that student profile (1–10 scale).  Results are written
to a JSONL file; already-scored triples are skipped so the run is resumable.

Typical usage (run from the project root):
    uv run python -m rl.scorer --config configs/phase3_ropg_kd.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.llm import OpenAICompatClient

logger = logging.getLogger(__name__)

# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert Persian-language educational assessor. "
    "You will be shown a question, a student profile, and a retrieved passage. "
    "Rate how useful this passage is for answering the question for a student "
    "with that specific profile, on a scale from 1 to 10.\n\n"
    "Scoring rubric:\n"
    "  10 — Directly answers the question in a way that perfectly fits the student's "
    "comprehension level, prior knowledge, learning goal, and preferred explanation style.\n"
    "   7 — Mostly relevant and fits the student well, but missing one dimension.\n"
    "   4 — Contains some relevant information but poorly matched to the student's needs.\n"
    "   1 — Irrelevant or actively misleading for this student.\n\n"
    "Output ONLY a single integer between 1 and 10. No explanation."
)


def _judge_prompt(query: str, profile_rendered: str, doc_text: str) -> list[dict[str, str]]:
    user = (
        f"Student profile: {profile_rendered}\n\n"
        f"Question: {query}\n\n"
        f"Passage:\n{doc_text}\n\n"
        "Utility score (1–10):"
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse_score(raw: str) -> float | None:
    """Extract a float in [1, 10] from the judge's raw response."""
    stripped = raw.strip()
    for token in stripped.split():
        try:
            val = float(token)
            if 1.0 <= val <= 10.0:
                return val
        except ValueError:
            continue
    return None


# ── Data loading ──────────────────────────────────────────────────────────────


def load_questions(exam_dir: str | Path) -> list[dict]:
    """Load all questions from raw exam JSONs in *exam_dir*."""
    exam_dir = Path(exam_dir)
    questions: list[dict] = []
    for path in sorted(exam_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        for q in data.get("questions", []):
            stem = q.get("stem", "").strip()
            if stem:
                questions.append({"source": path.stem, "id": q.get("id", ""), "stem": stem})
    return questions


# ── Resumable JSONL writer ────────────────────────────────────────────────────


def _load_scored(output_path: Path) -> set[tuple[str, str, str, int]]:
    """Return the set of (source, question_id, persona_id, doc_idx) already scored."""
    if not output_path.exists():
        return set()
    scored: set[tuple[str, str, str, int]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            scored.add((rec["source"], rec["question_id"], rec["persona_id"], rec["doc_idx"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return scored


# ── Main scoring loop ─────────────────────────────────────────────────────────


def run_scorer(config: dict) -> None:
    from rag.dense_store import DenseStore
    from rag.embedder import BGE_M3Embedder

    scorer_cfg = config["scorer"]
    output_path = Path(scorer_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_k: int = scorer_cfg.get("top_k", config["retrieval"]["top_k"])
    exam_dir = Path(scorer_cfg.get("exam_dir", "data/exams/raw"))

    # Build / open the dense index (read-only; we never add chunks here).
    emb_cfg = config.get("embedder", {})
    embedder = BGE_M3Embedder(
        model_name=emb_cfg.get("model", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cpu"),
    )
    store = DenseStore(
        index_path=config["knowledge_base"]["index_path"],
        meta_path=config["knowledge_base"]["meta_path"],
        embedder=embedder,
    )

    judge_cfg = scorer_cfg.get("judge_llm", config["llm"])
    judge = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=judge_cfg["model"],
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 10),
    )

    questions = load_questions(exam_dir)
    personas = train_personas()
    already_scored = _load_scored(output_path)

    total = len(questions) * len(personas) * top_k
    done = len(already_scored)
    logger.info(
        "Questions: %d | Personas: %d | Top-K: %d | Total triples: %d | Already done: %d",
        len(questions),
        len(personas),
        top_k,
        total,
        done,
    )

    with output_path.open("a", encoding="utf-8") as out_f:
        for q in questions:
            hits = store.search(q["stem"], top_k)
            for persona in personas:
                profile_rendered = render_profile(persona.id)
                for doc_idx, hit in enumerate(hits):
                    key = (q["source"], q["id"], persona.id, doc_idx)
                    if key in already_scored:
                        continue
                    messages = _judge_prompt(q["stem"], profile_rendered, hit.raw_text)
                    try:
                        raw = judge.chat(messages)
                        score = _parse_score(raw)
                    except Exception as exc:
                        logger.warning("Judge call failed: %s", exc)
                        score = None

                    record = {
                        "source": q["source"],
                        "question_id": q["id"],
                        "query": q["stem"],
                        "persona_id": persona.id,
                        "doc_idx": doc_idx,
                        "doc_text": hit.raw_text,
                        "doc_source": hit.source,
                        "score": score,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    already_scored.add(key)

    store.close()
    logger.info("Scoring complete. Output: %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Score (query, persona, doc) triples for ROPG-KD."
    )
    parser.add_argument("--config", required=True, help="Path to phase3 YAML config.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_scorer(config)


if __name__ == "__main__":
    main()
