"""
evaluate_embedding.py — Evaluate a fine-tuned Qwen3-Embedding checkpoint.

Supports base models and LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel  # <-- new import


# =============================================================================
# Model loading + encoding
# =============================================================================

def load_model(model_path: str, device: str, lora_path: str = None):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)

    # Apply LoRA if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # optional, merges weights into base model
        # Or keep as PeftModel; either works
        # model = model.to(device)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Mean-pool + L2 normalize."""
    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**tokens)
        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# =============================================================================
# Data loaders
# =============================================================================

def load_triplets(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            items.append({
                "query": obj["query"],
                "positive": obj["positive"],
            })
    return items


def load_pairs(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            items.append({
                "query": obj["query"],
                "positive": obj["positive"],
            })
    return items


def load_scored(path: str) -> List[Dict[str, Any]]:
    """Group scored rows by query.
    Supports both original format (documents, scores) and new format (docs with text and teacher_score).
    """
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    order: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            q = obj.get("query")
            if q is None:
                continue

            # Determine document list
            if "docs" in obj and isinstance(obj["docs"], list):
                # New format: docs is list of objects with "text" and "teacher_score"
                for doc in obj["docs"]:
                    text = doc.get("text")
                    score = float(doc.get("teacher_score", 0.0))
                    if text is not None:
                        grouped[q].append((text, score))
            elif "documents" in obj and "scores" in obj:
                # Old format: separate lists
                docs = obj["documents"]
                scores = obj["scores"]
                for doc, score in zip(docs, scores):
                    grouped[q].append((doc, float(score)))
            else:
                raise ValueError("Unknown scored format: expected 'docs' or 'documents' and 'scores' keys.")

            if q not in order:
                order.append(q)

    return [
        {"query": q, "documents": [d for d, _ in grouped[q]], "scores": [s for _, s in grouped[q]]}
        for q in order
    ]


# =============================================================================
# Metrics
# =============================================================================

def recall_at_k(scores: np.ndarray, top_k: int) -> Dict[str, float]:
    Q = scores.shape[0]
    results: Dict[str, float] = {}
    for k in range(1, top_k + 1):
        hits = 0
        for i in range(Q):
            topk_idx = np.argsort(scores[i])[::-1][:k]
            if i in topk_idx:
                hits += 1
        results[f"Recall@{k}"] = hits / max(Q, 1)
    return results


def mrr(scores: np.ndarray) -> float:
    Q = scores.shape[0]
    total = 0.0
    for i in range(Q):
        ranked = np.argsort(scores[i])[::-1]
        rank_pos = np.where(ranked == i)[0]
        if len(rank_pos) > 0:
            total += 1.0 / (rank_pos[0] + 1)
    return total / max(Q, 1)


def ndcg_at_k(gold_ranks: List[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gold_ranks[:k]):
        dcg += g / math.log2(i + 2)
    ideal = sorted(gold_ranks, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Evaluation entry points
# =============================================================================

def evaluate_triplets_or_pairs(
    model_path: str,
    test_path: str,
    device: str,
    top_k: int,
    lora_path: str = None,
) -> Dict[str, float]:
    model, tokenizer = load_model(model_path, device, lora_path)
    items = load_triplets(test_path) if test_path.endswith("_triplets.jsonl") else load_pairs(test_path)

    queries = [it["query"] for it in items]
    positives = [it["positive"] for it in items]

    print(f"Encoding {len(queries)} queries...")
    q_embs = encode_texts(model, tokenizer, queries, device)
    print(f"Encoding {len(positives)} positives...")
    p_embs = encode_texts(model, tokenizer, positives, device)

    scores = q_embs @ p_embs.T
    results = recall_at_k(scores, top_k)
    results["MRR"] = mrr(scores)
    return results


def evaluate_scored(
    model_path: str,
    test_path: str,
    device: str,
    top_k: int,
    lora_path: str = None,
) -> Dict[str, float]:
    model, tokenizer = load_model(model_path, device, lora_path)
    items = load_scored(test_path)

    print(f"Evaluating {len(items)} queries (each with its own candidate pool)...")

    recall_hits = {k: 0 for k in range(1, top_k + 1)}
    ndcg_sum = 0.0
    n = 0

    for it in items:
        q = it["query"]
        docs = it["documents"]
        gold = it["scores"]
        if len(docs) < 2:
            continue

        q_emb = encode_texts(model, tokenizer, [q], device)
        d_emb = encode_texts(model, tokenizer, docs, device)
        sims = (q_emb @ d_emb.T)[0]

        max_gold = max(gold)
        positive_indices = [i for i, s in enumerate(gold) if abs(s - max_gold) < 1e-6]

        ranked = np.argsort(sims)[::-1]
        for k in range(1, top_k + 1):
            topk_idx = ranked[:k]
            if any(p in topk_idx for p in positive_indices):
                recall_hits[k] += 1

        gold_arr = np.array(gold, dtype=float)
        gold_ranks = gold_arr[ranked].tolist()
        ndcg_sum += ndcg_at_k(gold_ranks, top_k)
        n += 1

    results: Dict[str, float] = {}
    for k in range(1, top_k + 1):
        results[f"Recall@{k}"] = recall_hits[k] / max(n, 1)
    results[f"NDCG@{top_k}"] = ndcg_sum / max(n, 1)
    return results


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Qwen3-Embedding checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base model checkpoint dir.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to the LoRA adapter directory (optional).")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to val_triplets.jsonl / val.jsonl / val_scored.jsonl.")
    parser.add_argument("--format", choices=["triplets", "pairs", "scored"], default=None,
                        help="Override format auto-detection (defaults inferred from filename).")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmt = args.format
    if fmt is None:
        if args.test_data.endswith("_triplets.jsonl"):
            fmt = "triplets"
        elif args.test_data.endswith("_scored.jsonl"):
            fmt = "scored"
        else:
            # Try to detect based on content (not perfect)
            with open(args.test_data, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                try:
                    obj = json.loads(first_line)
                    if "docs" in obj or ("documents" in obj and "scores" in obj):
                        fmt = "scored"
                    elif "positive" in obj:
                        fmt = "triplets"  # or pairs
                    else:
                        fmt = "pairs"
                except:
                    fmt = "pairs"

    print(f"Device : {device}")
    print(f"Model  : {args.model_path}")
    if args.lora_path:
        print(f"LoRA   : {args.lora_path}")
    print(f"Test   : {args.test_data}")
    print(f"Format : {fmt}")
    print()

    if fmt in ("triplets", "pairs"):
        results = evaluate_triplets_or_pairs(
            args.model_path, args.test_data, device, args.top_k, args.lora_path
        )
    else:
        results = evaluate_scored(
            args.model_path, args.test_data, device, args.top_k, args.lora_path
        )

    print("=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric:<14} {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()