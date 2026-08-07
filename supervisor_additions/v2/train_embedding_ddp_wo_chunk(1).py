#!/usr/bin/env python3
"""
train_embedding_ddp.py – Fine‑tune Qwen3‑Embedding with multi‑GPU (DDP) and LoRA support.

Usage (Kaggle with 2 GPUs):
    torchrun --nproc_per_node=2 train_embedding_ddp.py \
        --model_name Qwen/Qwen3-Embedding-0.6B \
        --train_data ./ropg_dataset \
        --output_dir ./qwen3-embed-ropg \
        --mode hard_neg \
        --format triplets \
        --num_epochs 3 \
        --batch_size 4 \
        --max_length 4096 \
        --learning_rate 2e-5 \
        --eval_top_k 5 \
        --best_metric recall@1 \
        --use_lora \
        --use_amp
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Persona renderer – replace with your own implementation if needed
def render_profile(persona_id: str) -> str:
    # Placeholder – adapt to your actual profile loader
    return f"[Persona: {persona_id}]"


# =============================================================================
# Datasets
# =============================================================================

class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str, max_negatives: int = 4) -> None:
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                negs = obj.get("negatives", [])[:max_negatives]
                if not negs:
                    continue
                query = obj["query"]
                if "persona_id" in obj:
                    persona_text = render_profile(obj["persona_id"])
                    query = f"{persona_text}\n\n{query}"
                self.items.append({
                    "query": query,
                    "positive": obj["positive"],
                    "negatives": negs,
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class PairDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self.items: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                query = obj["query"]
                if "persona_id" in obj:
                    persona_text = render_profile(obj["persona_id"])
                    query = f"{persona_text}\n\n{query}"
                self.items.append({
                    "query": query,
                    "positive": obj["positive"],
                    "negative": obj["negative"],
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


class ScoredDataset(Dataset):
    def __init__(self, jsonl_path: str, max_documents: int = 20) -> None:
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        order: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
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
                    continue

        self.items: List[Dict[str, Any]] = []
        for q in order:
            docs_scores = grouped[q][:max_documents]
            if len(docs_scores) < 2:
                continue
            docs, scores = zip(*docs_scores)
            self.items.append({
                "query": q,
                "documents": list(docs),
                "scores": list(scores),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# =============================================================================
# Model with LoRA support
# =============================================================================

class QwenEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, use_gradient_checkpointing: bool = True,
                 use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.05):
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        )
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is not installed. Install with: pip install peft")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model = get_peft_model(self.model, lora_config)
            if dist.get_rank() == 0:
                self.model.print_trainable_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / denom
        return F.normalize(emb, p=2, dim=1)


# =============================================================================
# Loss functions
# =============================================================================

def mnrl_loss(query_emb: torch.Tensor, pos_emb: torch.Tensor,
              neg_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    q = query_emb / temperature
    p = pos_emb / temperature
    n = neg_emb / temperature
    pos_score = (q * p).sum(dim=-1, keepdim=True)
    neg_scores = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1)
    all_scores = torch.cat([pos_score, neg_scores], dim=-1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(all_scores, labels)


def kd_loss(student_scores: torch.Tensor, gold_scores: torch.Tensor,
            temperature: float = 1.0) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_scores / temperature, dim=-1)
    gold_probs = F.softmax(gold_scores / temperature, dim=-1)
    return F.kl_div(student_logprobs, gold_probs, reduction="batchmean")


# =============================================================================
# Collators
# =============================================================================

def collate_triplets(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    queries = [x["query"] for x in batch]
    positives = [x["positive"] for x in batch]
    max_negs = max(len(x["negatives"]) for x in batch)

    negs_flat: List[str] = []
    for x in batch:
        negs_flat.extend(x["negatives"])
        negs_flat.extend([""] * (max_negs - len(x["negatives"])))

    q = tokenizer(queries, max_length=max_length, padding=True,
                  truncation=True, return_tensors="pt")
    p = tokenizer(positives, max_length=max_length, padding=True,
                  truncation=True, return_tensors="pt")
    n = tokenizer(negs_flat, max_length=max_length, padding=True,
                  truncation=True, return_tensors="pt")

    B = len(batch)
    return {
        "q_ids": q["input_ids"],
        "q_mask": q["attention_mask"],
        "p_ids": p["input_ids"],
        "p_mask": p["attention_mask"],
        "n_ids": n["input_ids"].view(B, max_negs, -1),
        "n_mask": n["attention_mask"].view(B, max_negs, -1),
    }


def collate_pairs(batch: List[Dict[str, str]], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    triplet_batch = [
        {"query": x["query"], "positive": x["positive"], "negatives": [x["negative"]]}
        for x in batch
    ]
    return collate_triplets(triplet_batch, tokenizer, max_length)


def collate_scored(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    queries = [x["query"] for x in batch]
    max_docs = max(len(x["documents"]) for x in batch)

    docs_flat: List[str] = []
    gold_scores: List[List[float]] = []
    doc_mask: List[List[bool]] = []
    for x in batch:
        n = len(x["documents"])
        docs_flat.extend(x["documents"])
        docs_flat.extend([""] * (max_docs - n))
        padded_scores = list(x["scores"]) + [-1e9] * (max_docs - n)
        gold_scores.append(padded_scores)
        doc_mask.append([True] * n + [False] * (max_docs - n))

    q = tokenizer(queries, max_length=max_length, padding=True,
                  truncation=True, return_tensors="pt")
    d = tokenizer(docs_flat, max_length=max_length, padding=True,
                  truncation=True, return_tensors="pt")

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
# Training loops (DDP-aware, only rank 0 prints)
# =============================================================================

def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_mnrl_epoch(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    temperature: float,
    use_amp: bool,
    log_every: int = 0,
    rank: int = 0,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Training (MNRL)", leave=False) if rank == 0 else loader
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

        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total / n:.4f}'})
            if log_every and n % log_every == 0:
                print(f"  Batch {n}: loss = {loss.item():.4f} (avg: {total / n:.4f})")

    if rank == 0:
        pbar.close()
    return total / max(n, 1)


def train_kd_epoch(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    temperature: float,
    use_amp: bool,
    log_every: int = 0,
    rank: int = 0,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Training (KD)", leave=False) if rank == 0 else loader
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

        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total / n:.4f}'})
            if log_every and n % log_every == 0:
                print(f"  Batch {n}: loss = {loss.item():.4f} (avg: {total / n:.4f})")

    if rank == 0:
        pbar.close()
    return total / max(n, 1)


@torch.no_grad()
def eval_mnrl_epoch(model: DDP, loader: DataLoader, device: torch.device,
                    temperature: float, use_amp: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
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
def eval_kd_epoch(model: DDP, loader: DataLoader, device: torch.device,
                  temperature: float, use_amp: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
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
# Evaluation helpers
# =============================================================================

def load_triplets_for_eval(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            query = obj["query"]
            if "persona_id" in obj:
                persona_text = render_profile(obj["persona_id"])
                query = f"{persona_text}\n\n{query}"
            items.append({"query": query, "positive": obj["positive"]})
    return items


def load_scored_for_eval(path: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    order: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
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
        {"query": q, "documents": [d for d, _ in grouped[q]], "scores": [s for _, s in grouped[q]]}
        for q in order
    ]


@torch.no_grad()
def encode_texts(
    model: DDP,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    model.eval()
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
        emb = model(tokens["input_ids"], tokens["attention_mask"])
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def recall_at_k(scores: np.ndarray, top_k: int) -> Dict[str, float]:
    Q = scores.shape[0]
    results = {}
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


def ndcg_at_k(gold_ranks: List[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gold_ranks[:k]):
        dcg += g / np.log2(i + 2)
    ideal = sorted(gold_ranks, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    model: DDP,
    tokenizer,
    test_path: str,
    fmt: str,
    device: str,
    top_k: int,
    batch_size: int = 32,
) -> Dict[str, float]:
    if fmt in ("triplets", "pairs"):
        items = load_triplets_for_eval(test_path)
        queries = [it["query"] for it in items]
        positives = [it["positive"] for it in items]

        q_embs = encode_texts(model, tokenizer, queries, device, batch_size)
        p_embs = encode_texts(model, tokenizer, positives, device, batch_size)
        scores = q_embs @ p_embs.T

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

            q_emb = encode_texts(model, tokenizer, [q], device, batch_size)
            d_emb = encode_texts(model, tokenizer, docs, device, batch_size)
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

        results = {}
        for k in range(1, top_k + 1):
            results[f"recall@{k}"] = recall_hits[k] / max(n, 1)
        results[f"ndcg@{top_k}"] = ndcg_sum / max(n, 1)
        return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding with multi-GPU support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./qwen3-embed-ropg")
    parser.add_argument("--mode", choices=["hard_neg", "reader_kd"], default="hard_neg")
    parser.add_argument("--format", choices=["triplets", "pairs"], default="triplets")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512, help="Context length (max tokens).")
    parser.add_argument("--max_negatives", type=int, default=4)
    parser.add_argument("--max_documents", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # Evaluation
    parser.add_argument("--eval_top_k", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--best_metric", type=str, default=None)

    # Logging
    parser.add_argument("--log_every", type=int, default=0)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Distributed training – `--local_rank` is automatically set by torchrun
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by launcher).")

    args = parser.parse_args()

    # ---- FIXED: read LOCAL_RANK from environment if not provided ----
    if args.local_rank == -1:
        local_rank_env = os.environ.get('LOCAL_RANK')
        if local_rank_env is not None:
            args.local_rank = int(local_rank_env)

    # ---- Initialize distributed process group ----
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed (same for all ranks)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if rank == 0:
        print(f"Device : {device}")
        print(f"Model  : {args.model_name}")
        print(f"Mode   : {args.mode}")
        print(f"Format : {args.format}")
        print(f"Max length: {args.max_length}")
        print(f"LoRA    : {args.use_lora}")
        print(f"World size: {world_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # ---- Build datasets ----
    train_data = Path(args.train_data)
    if args.mode == "hard_neg":
        if args.format == "triplets":
            train_path = train_data / "train_triplets.jsonl"
            val_path = train_data / "val_triplets.jsonl"
            train_ds = TripletDataset(str(train_path), args.max_negatives)
            val_ds = TripletDataset(str(val_path), args.max_negatives) if val_path.exists() else None
            collate_fn = lambda b: collate_triplets(b, tokenizer, args.max_length)
            eval_fmt = "triplets"
        else:
            train_path = train_data / "train.jsonl"
            val_path = train_data / "val.jsonl"
            train_ds = PairDataset(str(train_path))
            val_ds = PairDataset(str(val_path)) if val_path.exists() else None
            collate_fn = lambda b: collate_pairs(b, tokenizer, args.max_length)
            eval_fmt = "pairs"
    else:  # reader_kd
        train_path = train_data / "train_scored.jsonl"
        val_path = train_data / "val_scored.jsonl"
        train_ds = ScoredDataset(str(train_path), args.max_documents)
        val_ds = ScoredDataset(str(val_path), args.max_documents) if val_path.exists() else None
        collate_fn = lambda b: collate_scored(b, tokenizer, args.max_length)
        eval_fmt = "scored"

    if rank == 0:
        print(f"Train samples: {len(train_ds)}")
        if val_ds:
            print(f"Val samples  : {len(val_ds)}")
        else:
            print("Warning: No validation file found; evaluation will be skipped.")

    # Distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                   num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
        if val_ds else None
    )

    # ---- Model ----
    model = QwenEmbeddingModel(
        args.model_name,
        use_gradient_checkpointing=True,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    # Wrap with DDP if we are in distributed mode
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if rank == 0:
        print(f"Steps/epoch  : {steps_per_epoch}")
        print(f"Total steps  : {total_steps}")
        print(f"Warmup steps : {warmup_steps}")
        print()

    # Choose train/eval functions
    if args.mode == "hard_neg":
        train_fn = train_mnrl_epoch
        eval_loss_fn = eval_mnrl_epoch
    else:
        train_fn = train_kd_epoch
        eval_loss_fn = eval_kd_epoch

    if args.best_metric is None:
        best_metric_name = "val_loss"
        best_metric_goal = "min"
    else:
        best_metric_name = args.best_metric
        best_metric_goal = "max"

    if rank == 0:
        print(f"Best model selection based on: {best_metric_name} (goal: {best_metric_goal})")
        print()

    best_val_loss = float("inf")
    best_metric_value = -float("inf") if best_metric_goal == "max" else float("inf")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.output_dir) / "checkpoint-best"

    # ---- Training loop ----
    for epoch in range(1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device,
            args.temperature, args.use_amp,
            log_every=args.log_every,
            rank=rank,
        )

        val_str = ""
        if val_loader:
            val_loss = eval_loss_fn(
                model, val_loader, device,
                args.temperature, args.use_amp,
            )
            val_str = f" | Val Loss: {val_loss:.4f}"

            # Only rank 0 computes retrieval metrics and saves checkpoints
            if rank == 0:
                eval_metrics = evaluate_retrieval(
                    model, tokenizer, str(val_path), eval_fmt, device,
                    args.eval_top_k, args.eval_batch_size
                )
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
                val_str += f" | {metric_str}"

                # Save best checkpoint
                if args.best_metric is None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.module.model.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        val_str += " *best (loss)*"
                else:
                    current_metric = eval_metrics.get(best_metric_name)
                    if current_metric is not None:
                        if best_metric_goal == "max" and current_metric > best_metric_value:
                            best_metric_value = current_metric
                            model.module.model.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)
                            val_str += f" *best ({best_metric_name}: {current_metric:.4f})*"
                        elif best_metric_goal == "min" and current_metric < best_metric_value:
                            best_metric_value = current_metric
                            model.module.model.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)
                            val_str += f" *best ({best_metric_name}: {current_metric:.4f})*"
                    else:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            model.module.model.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)
                            val_str += " *best (loss fallback)*"

        if rank == 0:
            print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}{val_str}")

        # Periodic checkpoint (only rank 0)
        if rank == 0 and epoch % max(1, args.num_epochs // 3) == 0:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
            model.module.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # Save final (only rank 0)
    if rank == 0:
        final_dir = Path(args.output_dir) / "checkpoint-final"
        model.module.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print()
        print("Training complete!")
        print(f"  Best model : {save_dir}")
        print(f"  Final model: {final_dir}")

    # Cleanup distributed
    if args.local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()