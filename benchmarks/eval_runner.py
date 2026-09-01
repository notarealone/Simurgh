from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
for _p in (str(SRC_DIR), str(BENCHMARKS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa  (deliberate mid-file imports; sys.path must be set first)
from judge import GeminiJudge  # noqa: E402
from personalization.profiles import PERSONAS, render_profile  # noqa: E402
from rag.pipeline import PersonaRAG  # noqa: E402
from rag.prompts import build_mcq_rag_prompt  # noqa: E402


def load_questions(questions_dir: str) -> list[dict]:
    """Load all MCQ questions from JSON exam files in *questions_dir*."""
    questions = []
    qdir = Path(questions_dir)
    for json_path in sorted(qdir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            exam = json.load(f)
        passages = {p["group_id"]: p["text"] for p in exam.get("passages", [])}
        for q in exam.get("questions", []):
            if q.get("type") != "mcq":
                continue
            group_id = q.get("group_id")
            passage_text = passages.get(group_id) if group_id else None
            correct_answer_index = q["answer"] - 1  # 1-based → 0-based
            questions.append({
                "source_file": json_path.name,
                "exam_title": exam["exam_title"],
                "question_id": q["id"],
                "stem": q["stem"],
                "options": q["options"],
                "correct_answer_index": correct_answer_index,
                "correct_answer_text": q["options"][correct_answer_index],
                "group_id": group_id,
                "passage_text": passage_text,
            })
    return questions


def build_retrieval_query(q: dict) -> str:
    """Build retrieval query from question stem (+ passage if group_id is set)."""
    if q["passage_text"]:
        return q["passage_text"] + "\n" + q["stem"]
    return q["stem"]


def build_pipeline(step_cfg: dict, eval_cfg: dict, seed: int) -> PersonaRAG:
    """Build a PersonaRAG pipeline from step and eval configs."""
    config = {
        "seed": seed,
        "knowledge_base": {
            "index_path": step_cfg["index_path"],
            "meta_path": step_cfg["meta_path"],
        },
        "embedder": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "device": "cuda",
            "batch_size": 32,
            "adapter_path": step_cfg.get("embedder_adapter_path"),
        },
        "chunking": {
            "chunk_size": eval_cfg["chunking"]["chunk_size"],
            "chunk_overlap": eval_cfg["chunking"]["chunk_overlap"],
        },
        "retrieval": {
            "top_k": eval_cfg["retrieval"]["top_k"],
        },
        "llm": {
            "model": step_cfg["generator_model"],
            "temperature": 0.2,
            "max_tokens": 800,
            "prompt_variant": "en",
        },
        "rewriter": {
            "enabled": step_cfg.get("rewriter_type", "none") != "none",
        },
    }
    if step_cfg.get("rewriter_type") == "dpo":
        config["rewriter"].update(
            {
                "type": "dpo",
                "model": step_cfg.get("dpo_model", "Qwen/Qwen3-4B"),
                "adapter_path": step_cfg.get("dpo_adapter_path"),
                "device": "cuda",
                "max_seq_length": step_cfg.get("dpo_max_seq_length", 768),
                "generation": {"max_new_tokens": 224, "do_sample": False},
            }
        )


def run_eval(eval_config_path: str) -> None:
    cfg_path = Path(eval_config_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_questions = load_questions(cfg["questions_dir"])
    persona_ids = list(PERSONAS.keys())
    judge = GeminiJudge(
        model=cfg["judge"]["model"],
        max_retries=cfg["judge"].get("max_retries", 3),
    )
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus for logging
    corpus_path = Path(cfg["corpus_jsonl"])
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus_chunks = [json.loads(line) for line in f if line.strip()]
        print(f"Corpus: {len(corpus_chunks)} chunks loaded from {corpus_path}")
    else:
        print(f"Warning: corpus file not found at {corpus_path}")

    all_results: list[dict] = []

    for seed in cfg["seeds"]:
        random.seed(seed)
        for step_name, step_cfg in cfg["steps"].items():
            print(f"\n{'=' * 60}")
            print(f"Seed {seed} | Step: {step_cfg['name']}")
            print(f"Description: {step_cfg['description']}")
            print(f"{'=' * 60}")

            try:
                pipeline = build_pipeline(step_cfg, cfg, seed)
            except Exception as e:
                print(f"  ERROR building pipeline: {e}")
                continue

            step_results: list[dict] = []
            for q in all_questions:
                retrieval_query = build_retrieval_query(q)

                for persona_id in persona_ids:
                    profile_str = render_profile(persona_id)

                    # Rewrite query if rewriter exists
                    if pipeline.rewriter is not None:
                        rewritten_query = pipeline.rewriter.rewrite(profile_str, retrieval_query)
                    else:
                        rewritten_query = retrieval_query

                    # Search
                    instruction = profile_str
                    hits = pipeline.kb.search(
                        rewritten_query,
                        cfg["retrieval"]["top_k"],
                        instruction=instruction,
                    )

                    # Generate MCQ answer
                    messages = build_mcq_rag_prompt(
                        q["stem"],
                        q["options"],
                        hits,
                        variant="en",
                    )
                    answer = pipeline.llm.chat(messages)

                    # Judge
                    scores = judge.score(
                        persona=profile_str,
                        query=q["stem"],
                        rewritten_query=rewritten_query,
                        options=q["options"],
                        correct_option=q["correct_answer_text"],
                        hits=hits,
                        answer=answer,
                    )

                    record = {
                        "seed": seed,
                        "step": step_cfg["name"],
                        "persona_id": persona_id,
                        "is_test_persona": persona_id == cfg["test_persona"],
                        "question_id": q["question_id"],
                        "source_file": q["source_file"],
                        "exam_title": q["exam_title"],
                        "stem": q["stem"],
                        "options": q["options"],
                        "correct_answer_text": q["correct_answer_text"],
                        "retrieval_query": retrieval_query,
                        "rewritten_query": rewritten_query,
                        "hit_sources": [h.source for h in hits],
                        "answer": answer,
                        **scores,
                    }
                    step_results.append(record)
                    all_results.append(record)

                    mean_score = (
                        (scores.get("persona_alignment", 0) or 0)
                        + (scores.get("pedagogical_quality", 0) or 0)
                        + (scores.get("faithfulness", 0) or 0)
                        + (scores.get("overall", 0) or 0)
                    ) / 4.0
                    print(
                        f"  {q['question_id']} | {persona_id:10s} | "
                        f"overall={scores.get('overall', -1)} acc={scores.get('accuracy', -1)} "
                        f"mean={mean_score:.1f}"
                    )

            # Write step results
            out_path = output_dir / f"{step_name}_seed{seed}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in step_results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  → {len(step_results)} results → {out_path}")

            pipeline.close()

    # Write aggregate summary CSV
    summary_path = output_dir / "summary.csv"
    step_names = sorted(set(r["step"] for r in all_results))
    all_personas = sorted(set(r["persona_id"] for r in all_results))

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step_name", "persona_id", "is_test_persona",
            "persona_alignment_mean", "pedagogical_quality_mean",
            "faithfulness_mean", "overall_mean", "accuracy_mean", "n",
        ])
        for step_name in step_names:
            for pid in all_personas:
                subset = [r for r in all_results if r["step"] == step_name and r["persona_id"] == pid]
                n = len(subset)
                if n == 0:
                    continue
                is_test = subset[0]["is_test_persona"]

                def _mean(key):
                    vals = [r.get(key, 0) or 0 for r in subset if r.get(key, -1) != -1]
                    return sum(vals) / len(vals) if vals else -1

                writer.writerow([
                    step_name, pid, is_test,
                    round(_mean("persona_alignment"), 2),
                    round(_mean("pedagogical_quality"), 2),
                    round(_mean("faithfulness"), 2),
                    round(_mean("overall"), 2),
                    round(_mean("accuracy"), 2),
                    n,
                ])

    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eval_ablation.yaml"
    run_eval(config_path)
