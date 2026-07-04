"""Generate DPO rewriter preference pairs: N rewrites per (query, persona) ranked by LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import yaml

from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL
from personalization.profiles import render_profile, train_personas
from rag.llm import OpenAICompatClient
from rag.rewriter import PromptedRewriter

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are an expert Persian language tutor evaluating query rewrites for a RAG retrieval system."


def _build_judge_messages(
    persona_rendered: str, original_query: str, rewrite: str
) -> list[dict[str, str]]:
    user = (
        f"A student with the following profile is searching for study material:\n"
        f"Profile: {persona_rendered}\n\n"
        f"Original exam question: {original_query}\n\n"
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
    cleaned = response.strip()
    match = re.search(r"[\d.]+", cleaned)
    if match is None:
        logger.warning("Could not extract score from judge response: %r", response[:200])
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


def _index_questions(questions_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for fpath in sorted(questions_dir.glob("*.json")):
        result[fpath.stem] = fpath
    logger.info("Found %d question files in %s", len(result), questions_dir)
    return result


def _load_question_stem(exam_file: Path, qid: str) -> str:
    data = json.loads(exam_file.read_text(encoding="utf-8"))
    for q in data.get("questions", []):
        if q.get("id") == qid:
            return q["stem"]
    raise KeyError(f"Question {qid!r} not found in {exam_file}")


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

    temperatures: list[float] = rewriter_cfg.get("temperatures", [0.3, 0.5, 0.7, 0.9, 1.1, 1.3])

    questions_dir = Path(data_cfg["questions_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_map = _index_questions(questions_dir)
    if not questions_map:
        logger.error("No question files found in %s", questions_dir)
        return

    persona_ids = [p.id for p in train_personas()]

    judge_client = OpenAICompatClient(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=judge_cfg["model"],
        temperature=judge_cfg["temperature"],
        max_tokens=judge_cfg["max_tokens"],
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
            for exam_stem, qid in qid_pairs:
                exam_file = questions_map.get(exam_stem)
                if exam_file is None:
                    logger.warning("Exam file not found for stem %r", exam_stem)
                    continue

                try:
                    query = _load_question_stem(exam_file, qid)
                except (KeyError, json.JSONDecodeError) as e:
                    logger.warning("Skipping %s:%s: %s", exam_stem, qid, e)
                    continue

                logger.info(
                    "Processing %s:%s — %d chars — %s",
                    exam_stem,
                    qid,
                    len(query),
                    split_name,
                )

                best_for_persona: dict[str, str] = {}
                best_score_for_persona: dict[str, float] = {}

                for persona_id in persona_ids:
                    persona_rendered = render_profile(persona_id)
                    candidates: list[tuple[str, float]] = []

                    for temp in temperatures:
                        rewriter_client = OpenAICompatClient(
                            base_url=OPENAI_BASE_URL,
                            api_key=OPENAI_API_KEY,
                            model=rewriter_cfg["model"],
                            temperature=temp,
                            max_tokens=rewriter_cfg["max_tokens"],
                        )
                        rewriter = PromptedRewriter(rewriter_client)

                        try:
                            rewrite = rewriter.rewrite(persona_rendered, query)
                        except Exception:
                            logger.warning(
                                "Rewrite failed for %s:%s persona=%s temp=%.1f",
                                exam_stem,
                                qid,
                                persona_id,
                                temp,
                                exc_info=True,
                            )
                            continue

                        try:
                            messages = _build_judge_messages(persona_rendered, query, rewrite)
                            response = judge_client.chat(messages)
                            score = _parse_score(response)
                        except Exception:
                            logger.warning(
                                "Judge failed for %s:%s persona=%s temp=%.1f",
                                exam_stem,
                                qid,
                                persona_id,
                                temp,
                                exc_info=True,
                            )
                            score = 0.0

                        candidates.append((rewrite, score))

                    if not candidates:
                        logger.error(
                            "No candidates for %s:%s persona=%s — skipping persona",
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

                    record = {
                        "persona_id": persona_id,
                        "query": query,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
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

                        record_ab = {
                            "persona_id": pid_a,
                            "query": query,
                            "chosen": best_for_persona[pid_a],
                            "rejected": best_for_persona[pid_b],
                        }
                        fh.write(json.dumps(record_ab, ensure_ascii=False) + "\n")
                        records_written += 1

                        record_ba = {
                            "persona_id": pid_b,
                            "query": query,
                            "chosen": best_for_persona[pid_b],
                            "rejected": best_for_persona[pid_a],
                        }
                        fh.write(json.dumps(record_ba, ensure_ascii=False) + "\n")
                        records_written += 1

        logger.info("Wrote %d records to %s", records_written, output_path)


if __name__ == "__main__":
    main()
