"""ROPG-KD: fine-tune the Qwen3-Embedding-0.6B encoder via knowledge distillation from LLM judge scores.

Stage 1 of Simurgh's two-stage training:
  1. Load (query, persona, docs) groups scored by the offline LLM judge
     (notebooks/gen_ropg_data.ipynb, mirroring configs/datagen_ropg.yaml),
     each doc carrying a teacher_score in [0, 1].
  2. Fine-tune the Qwen3-Embedding-0.6B encoder to minimise the listwise
     KL divergence between its similarity distribution over the group's docs and the
     teacher's softmax utility distribution.
  3. Validate every epoch with Recall@K / MRR per persona and checkpoint
     on the best metric.

Supports two training modes:
  - hard_neg: Multiple Negatives Ranking Loss (MNRL)
  - reader_kd: KL-distillation against continuous reader scores

Runs on GPU (Kaggle / university cluster). Install the training extras first:
    uv sync --extra embedding --extra training

Typical usage:
    uv run python -m rl.ropg_kd --config configs/train_ropg.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from personalization.profiles import render_profile

logger = logging.getLogger(__name__)


# =============================================================================
# Datasets — now with persona conditioning
# =============================================================================


class TripletDataset(Dataset):
    """Reads {query, positive, negatives: [...]} lines.
    If 'persona_id' is present, prepend the rendered profile to the query.
    """

    def __init__(self, jsonl_path: str, max_negatives: int = 4) -> None:
        self.items: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                negs = obj.get("negatives", [])[:max_negatives]
                if not negs:
                    continue
                query = obj["query"]
                if "persona_id" in obj:
                    persona_text = render_profile(obj["persona_id"])
                    query = f"{persona_text}\n\n{query}"
                self.items.append(
                    {
                        "query": query,
                        "positive": obj["positive"],
                        "negatives": negs,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class PairDataset(Dataset):
    """Reads {query, positive, negative} lines (cartesian-expanded).
    If 'persona_id' is present, prepend the rendered profile to the query.
    """

    def __init__(self, jsonl_path: str) -> None:
        self.items: list[dict[str, str]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                query = obj["query"]
                if "persona_id" in obj:
                    persona_text = render_profile(obj["persona_id"])
                    query = f"{persona_text}\n\n{query}"
                self.items.append(
                    {
                        "query": query,
                        "positive": obj["positive"],
                        "negative": obj["negative"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.items[idx]


class ScoredDataset(Dataset):
    """Reads scored lines for KL-distillation.

    Supports two JSONL formats:
      1. Old: {"query": "...", "document": "...", "score": 0.5}
      2. New: {"query": "...", "persona_id": "...", "docs": [{"chunk_id": "...", "text": "...", "teacher_score": 0.3}, ...]}

    If 'persona_id' is present, prepend the rendered profile to the query.
    """

    def __init__(self, jsonl_path: str, max_documents: int = 20) -> None:
        grouped: dict[str, list[tuple[str, float]]] = {}
        order: list[str] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                q = obj["query"]
                if "persona_id" in obj:
                    persona_text = render_profile(obj["persona_id"])
                    q = f"{persona_text}\n\n{q}"
                if q not in grouped:
                    grouped[q] = []
                    order.append(q)

                if "docs" in obj:
                    for doc in obj["docs"]:
                        grouped[q].append((doc["text"], float(doc["teacher_score"])))
                elif "document" in obj and "score" in obj:
                    grouped[q].append((obj["document"], float(obj["score"])))
                else:
                    continue  # unknown format

        self.items: list[dict[str, Any]] = []
        for q in order:
            docs_scores = grouped[q][:max_documents]
            if len(docs_scores) < 2:
                continue
            docs, scores = zip(*docs_scores, strict=False)
            self.items.append(
                {
                    "query": q,  # already modified with persona
                    "documents": list(docs),
                    "scores": list(scores),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


# =============================================================================
# Model
# =============================================================================


class QwenEmbeddingModel(nn.Module):
    """Qwen3-Embedding wrapper with mean pooling + L2 normalization."""

    def __init__(self, model_name: str, use_gradient_checkpointing: bool = True):
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        )
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / denom
        return F.normalize(emb, p=2, dim=1)


# =============================================================================
# Losses
# =============================================================================


def mnrl_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    q = query_emb / temperature
    p = pos_emb / temperature
    n = neg_emb / temperature
    pos_score = (q * p).sum(dim=-1, keepdim=True)
    neg_scores = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1)
    all_scores = torch.cat([pos_score, neg_scores], dim=-1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(all_scores, labels)


def kd_loss(
    student_scores: torch.Tensor,
    gold_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_scores / temperature, dim=-1)
    gold_probs = F.softmax(gold_scores / temperature, dim=-1)
    return F.kl_div(student_logprobs, gold_probs, reduction="batchmean")


# =============================================================================
# Collators
# =============================================================================


def collate_triplets(
    batch: list[dict[str, Any]],
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    queries = [x["query"] for x in batch]
    positives = [x["positive"] for x in batch]
    max_negs = max(len(x["negatives"]) for x in batch)

    negs_flat: list[str] = []
    for x in batch:
        negs_flat.extend(x["negatives"])
        negs_flat.extend([""] * (max_negs - len(x["negatives"])))

    q = tokenizer(
        queries,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    p = tokenizer(
        positives,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    n = tokenizer(
        negs_flat,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    B = len(batch)
    return {
        "q_ids": q["input_ids"],
        "q_mask": q["attention_mask"],
        "p_ids": p["input_ids"],
        "p_mask": p["attention_mask"],
        "n_ids": n["input_ids"].view(B, max_negs, -1),
        "n_mask": n["attention_mask"].view(B, max_negs, -1),
    }


def collate_pairs(
    batch: list[dict[str, str]],
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    triplet_batch = [
        {"query": x["query"], "positive": x["positive"], "negatives": [x["negative"]]}
        for x in batch
    ]
    return collate_triplets(triplet_batch, tokenizer, max_length)


def collate_scored(
    batch: list[dict[str, Any]],
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    queries = [x["query"] for x in batch]
    max_docs = max(len(x["documents"]) for x in batch)

    docs_flat: list[str] = []
    gold_scores: list[list[float]] = []
    doc_mask: list[list[bool]] = []
    for x in batch:
        n = len(x["documents"])
        docs_flat.extend(x["documents"])
        docs_flat.extend([""] * (max_docs - n))
        padded_scores = list(x["scores"]) + [-1e9] * (max_docs - n)
        gold_scores.append(padded_scores)
        doc_mask.append([True] * n + [False] * (max_docs - n))

    q = tokenizer(
        queries,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    d = tokenizer(
        docs_flat,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    B = len(batch)
    return {
        "q_ids": q["input_ids"],
        "q_mask": q["attention_mask"],
        "d_ids": d["input_ids"].view(B, max_docs, -1),
        "d_mask": d["attention_mask"].view(B, max_docs, -1),
        "gold_scores": torch.tensor(gold_scores, dtype=torch.float32),
        "doc_mask": torch.tensor(doc_mask, dtype=torch.bool),
    }


# =============================================================================
# Training helpers
# =============================================================================


def _move(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# =============================================================================
# Training loops
# =============================================================================


def train_mnrl_epoch(
    model: QwenEmbeddingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = _move(batch, device)
        q_ids, q_mask = batch["q_ids"], batch["q_mask"]
        p_ids, p_mask = batch["p_ids"], batch["p_mask"]
        n_ids, n_mask = batch["n_ids"], batch["n_mask"]
        B, N, L = n_ids.shape

        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            q_emb = model(q_ids, q_mask)
            p_emb = model(p_ids, p_mask)
            n_flat = model(n_ids.view(B * N, L), n_mask.view(B * N, L)).view(B, N, -1)
            loss = mnrl_loss(q_emb, p_emb, n_flat, temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


def train_kd_epoch(
    model: QwenEmbeddingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = _move(batch, device)
        q_ids, q_mask = batch["q_ids"], batch["q_mask"]
        d_ids, d_mask = batch["d_ids"], batch["d_mask"]
        gold = batch["gold_scores"]
        B, D, L = d_ids.shape

        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            q_emb = model(q_ids, q_mask)
            d_emb = model(d_ids.view(B * D, L), d_mask.view(B * D, L)).view(B, D, -1)
            student_scores = torch.einsum("bh,bdh->bd", q_emb, d_emb)
            loss = kd_loss(student_scores, gold, temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


# =============================================================================
# Eval loops
# =============================================================================


@torch.no_grad()
def eval_mnrl_epoch(
    model: QwenEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move(batch, device)
        q_ids, q_mask = batch["q_ids"], batch["q_mask"]
        p_ids, p_mask = batch["p_ids"], batch["p_mask"]
        n_ids, n_mask = batch["n_ids"], batch["n_mask"]
        B, N, L = n_ids.shape
        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            q_emb = model(q_ids, q_mask)
            p_emb = model(p_ids, p_mask)
            n_flat = model(n_ids.view(B * N, L), n_mask.view(B * N, L)).view(B, N, -1)
            loss = mnrl_loss(q_emb, p_emb, n_flat, temperature)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_kd_epoch(
    model: QwenEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move(batch, device)
        q_ids, q_mask = batch["q_ids"], batch["q_mask"]
        d_ids, d_mask = batch["d_ids"], batch["d_mask"]
        gold = batch["gold_scores"]
        B, D, L = d_ids.shape
        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            q_emb = model(q_ids, q_mask)
            d_emb = model(d_ids.view(B * D, L), d_mask.view(B * D, L)).view(B, D, -1)
            student_scores = torch.einsum("bh,bdh->bd", q_emb, d_emb)
            loss = kd_loss(student_scores, gold, temperature)
        total += loss.item()
        n += 1
    return total / max(n, 1)


# =============================================================================
# Evaluation helpers (persona-aware)
# =============================================================================


def load_triplets_for_eval(path: str) -> list[dict[str, str]]:
    """Load only query and positive from a triplets/pairs JSONL, applying persona if present."""
    items: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            query = obj["query"]
            if "persona_id" in obj:
                persona_text = render_profile(obj["persona_id"])
                query = f"{persona_text}\n\n{query}"
            items.append(
                {
                    "query": query,
                    "positive": obj["positive"],
                }
            )
    return items


def load_scored_for_eval(path: str) -> list[dict[str, Any]]:
    """Group scored rows by query – each query has multiple (document, score) pairs.
    Applies persona to the query if present.
    """
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    order: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            q = obj["query"]
            if "persona_id" in obj:
                persona_text = render_profile(obj["persona_id"])
                q = f"{persona_text}\n\n{q}"
            if q not in grouped:
                order.append(q)
            if "docs" in obj:
                for doc in obj["docs"]:
                    grouped[q].append((doc["text"], float(doc["teacher_score"])))
            elif "document" in obj and "score" in obj:
                grouped[q].append((obj["document"], float(obj["score"])))
            else:
                continue
    return [
        {
            "query": q,
            "documents": [d for d, _ in grouped[q]],
            "scores": [s for _, s in grouped[q]],
        }
        for q in order
    ]


@torch.no_grad()
def encode_texts(
    model: QwenEmbeddingModel,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Encode texts using the trained wrapper model (mean-pool + L2)."""
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        emb = model(tokens["input_ids"], tokens["attention_mask"])
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# =============================================================================
# Metrics
# =============================================================================


def recall_at_k(scores: np.ndarray, top_k: int) -> dict[str, float]:
    """For each row i, the ground-truth index is also i."""
    Q = scores.shape[0]
    results: dict[str, float] = {}
    for k in range(1, top_k + 1):
        hits = 0
        for i in range(Q):
            topk_idx = np.argsort(scores[i])[::-1][:k]
            if i in topk_idx:
                hits += 1
        results[f"recall@{k}"] = hits / max(Q, 1)
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


def ndcg_at_k(gold_ranks: list[int], k: int) -> float:
    """NDCG@k for a single query (higher gold rank = more relevant)."""
    dcg = 0.0
    for i, g in enumerate(gold_ranks[:k]):
        dcg += g / math.log2(i + 2)
    ideal = sorted(gold_ranks, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    model: QwenEmbeddingModel,
    tokenizer,
    test_path: str,
    fmt: str,
    device: str,
    top_k: int,
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute retrieval metrics on the validation set.

    Args:
        fmt: "triplets", "pairs", or "scored"
    """
    if fmt in ("triplets", "pairs"):
        items = load_triplets_for_eval(test_path)
        queries = [it["query"] for it in items]
        positives = [it["positive"] for it in items]

        q_embs = encode_texts(model, tokenizer, queries, device, batch_size)
        p_embs = encode_texts(model, tokenizer, positives, device, batch_size)
        scores = q_embs @ p_embs.T  # [Q, P]

        results = recall_at_k(scores, top_k)
        results["mrr"] = mrr(scores)
        return results

    else:  # scored
        items = load_scored_for_eval(test_path)
        recall_hits = {k: 0 for k in range(1, top_k + 1)}
        ndcg_sum = 0.0
        n = 0

        for it in items:
            q = it["query"]
            docs = it["documents"]
            gold = it["scores"]
            if len(docs) < 2:
                continue

            q_emb = encode_texts(model, tokenizer, [q], device, batch_size)  # [1, H]
            d_emb = encode_texts(model, tokenizer, docs, device, batch_size)  # [D, H]
            sims = (q_emb @ d_emb.T)[0]  # [D]

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

        results: dict[str, float] = {}
        for k in range(1, top_k + 1):
            results[f"recall@{k}"] = recall_hits[k] / max(n, 1)
        results[f"ndcg@{top_k}"] = ndcg_sum / max(n, 1)
        return results


# =============================================================================
# Training entry point
# =============================================================================


def train(config: dict) -> None:
    """Main training routine driven by a YAML config dict."""
    # ── seed ────────────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── device & AMP ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config["training"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")

    logger.info("Device: %s | precision: %s | amp: %s", device, precision, use_amp)

    # ── paths ───────────────────────────────────────────────────────────────
    train_data = Path(config["data"]["train_data"])
    mode = config.get("mode", "reader_kd")
    fmt = config.get("format", "triplets")
    output_dir = Path(config.get("checkpoint_dir", "./ropg_kd_checkpoints"))

    if mode == "hard_neg":
        if fmt == "triplets":
            train_path = train_data / "train_triplets.jsonl"
            val_path = train_data / "val_triplets.jsonl"
            eval_fmt = "triplets"
        else:  # pairs
            train_path = train_data / "train_pairs.jsonl"
            val_path = train_data / "val_pairs.jsonl"
            eval_fmt = "pairs"
    else:  # reader_kd
        train_path = train_data / "train.jsonl"
        val_path = train_data / "val.jsonl"
        eval_fmt = "scored"

    # ── auto-derive triplets/pairs if missing ───────────────────────────────
    # The scored data (train.jsonl / val.jsonl) is always generated first.
    # Triplets and pairs are a coarse binarisation of that signal; they can be
    # derived on demand so the user doesn't have to re-run data generation just
    # to switch to hard_neg mode.
    if mode == "hard_neg" and not train_path.exists():
        scored_train = train_data / "train.jsonl"
        if not scored_train.exists():
            raise FileNotFoundError(
                f"Neither {train_path} nor {scored_train} found. "
                "Run data generation first (src/data/gen_ropg_data.py)."
            )
        logger.warning(
            "%s not found — deriving triplets/pairs from scored data in %s",
            train_path.name,
            train_data,
        )
        from data.gen_ropg_data import derive_triplets
        for scored_path in (train_data / "train.jsonl", train_data / "val.jsonl"):
            if scored_path.exists():
                derive_triplets(scored_path, train_data, config["training"].get("max_negatives", 4))

    # ── config-derived hyperparams ──────────────────────────────────────────
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    num_epochs = config["training"]["epochs"]
    warmup_ratio = config["training"].get("warmup_ratio", 0.1)
    weight_decay = config["training"].get("weight_decay", 0.01)
    temperature = config["training"]["temperature"]
    max_negatives = config["training"].get("max_negatives", 4)
    max_documents = config["training"].get("max_documents", 20)
    gradient_checkpointing = config["training"].get("gradient_checkpointing", True)
    num_workers = config["training"].get("num_workers", 2)

    model_name = config["embedder"]["model"]
    max_length = config["embedder"].get("max_seq_length", 512)

    best_metric = config["eval"].get("best_metric")
    eval_batch_size = config["eval"].get("eval_batch_size", 32)
    eval_top_k = config["eval"].get("top_k", 5)

    # ── tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ── datasets & loaders ──────────────────────────────────────────────────
    if mode == "hard_neg":
        if fmt == "triplets":
            train_ds = TripletDataset(str(train_path), max_negatives)
            val_ds = (
                TripletDataset(str(val_path), max_negatives)
                if val_path.exists()
                else None
            )
            collate_fn = lambda b: collate_triplets(b, tokenizer, max_length)  # noqa: E731
        else:
            train_ds = PairDataset(str(train_path))
            val_ds = PairDataset(str(val_path)) if val_path.exists() else None
            collate_fn = lambda b: collate_pairs(b, tokenizer, max_length)  # noqa: E731
    else:  # reader_kd
        train_ds = ScoredDataset(str(train_path), max_documents)
        val_ds = (
            ScoredDataset(str(val_path), max_documents)
            if val_path.exists()
            else None
        )
        collate_fn = lambda b: collate_scored(b, tokenizer, max_length)  # noqa: E731

    logger.info("Train samples: %d", len(train_ds))
    if val_ds:
        logger.info("Val samples  : %d", len(val_ds))
    else:
        logger.info("Warning: No validation file found; evaluation will be skipped.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        if val_ds
        else None
    )

    # ── model / optim / scheduler ───────────────────────────────────────────
    model = QwenEmbeddingModel(model_name, use_gradient_checkpointing=gradient_checkpointing).to(
        device
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters   : %d", total_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    logger.info("Steps/epoch  : %d", steps_per_epoch)
    logger.info("Total steps  : %d", total_steps)
    logger.info("Warmup steps : %d", warmup_steps)

    # ── best-metric tracking ────────────────────────────────────────────────
    if best_metric is None:
        best_metric_name = "val_loss"
        best_metric_goal = "min"
    else:
        best_metric_name = best_metric
        best_metric_goal = "max"

    logger.info(
        "Best model selection based on: %s (goal: %s)", best_metric_name, best_metric_goal
    )

    # ── pick train/eval functions ───────────────────────────────────────────
    if mode == "hard_neg":
        train_fn = train_mnrl_epoch
        eval_loss_fn = eval_mnrl_epoch
    else:
        train_fn = train_kd_epoch
        eval_loss_fn = eval_kd_epoch

    # ── checkpoint dirs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "checkpoint-best"

    best_val_loss = float("inf")
    best_metric_value = -float("inf") if best_metric_goal == "max" else float("inf")
    best_epoch = 0
    epoch_metrics: list[dict[str, Any]] = []

    # ── training loop ───────────────────────────────────────────────────────
    for epoch in tqdm(range(1, num_epochs + 1), desc="epochs"):
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            temperature,
            use_amp,
        )

        val_str = ""
        epoch_entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if val_loader and val_path.exists():
            # Validation loss
            val_loss = eval_loss_fn(
                model,
                val_loader,
                device,
                temperature,
                use_amp,
            )
            val_str = f" | Val Loss: {val_loss:.4f}"
            epoch_entry["val_loss"] = val_loss

            # Retrieval metrics on validation set
            eval_metrics = evaluate_retrieval(
                model,
                tokenizer,
                str(val_path),
                eval_fmt,
                str(device),
                eval_top_k,
                eval_batch_size,
            )
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
            val_str += f" | {metric_str}"
            epoch_entry.update(eval_metrics)

            # Best checkpoint selection
            if best_metric is None:
                # Use validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    best_epoch = epoch
                    val_str += " *best (loss)*"
            else:
                current_metric = eval_metrics.get(best_metric_name)
                if current_metric is not None:
                    improved = (
                        current_metric > best_metric_value
                        if best_metric_goal == "max"
                        else current_metric < best_metric_value
                    )
                    if improved:
                        best_metric_value = current_metric
                        model.model.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        best_epoch = epoch
                        val_str += (
                            f" *best ({best_metric_name}: {current_metric:.4f})*"
                        )
                else:
                    # Fallback to loss if metric not found
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.model.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        best_epoch = epoch
                        val_str += " *best (loss fallback)*"

        epoch_metrics.append(epoch_entry)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f%s",
            epoch,
            num_epochs,
            train_loss,
            val_str,
        )

        # Periodic checkpoint
        freq = max(1, num_epochs // 3)
        if epoch % freq == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch}"
            model.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Periodic checkpoint saved: %s", ckpt_dir)

    # ── final checkpoint ────────────────────────────────────────────────────
    final_dir = output_dir / "checkpoint-final"
    model.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ── training log ────────────────────────────────────────────────────────
    training_log = {
        "config": config,
        "seed": seed,
        "epoch_metrics": epoch_metrics,
        "best_epoch": best_epoch,
    }
    log_path = output_dir / "training_log.json"
    log_path.write_text(
        json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Training complete!")
    logger.info("  Best model : %s", save_dir)
    logger.info("  Final model: %s", final_dir)
    logger.info("  Training log: %s", log_path)


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="ROPG-KD: fine-tune Qwen3-Embedding encoder with KD."
    )
    parser.add_argument(
        "--config", required=True, help="Path to train_ropg.yaml."
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
