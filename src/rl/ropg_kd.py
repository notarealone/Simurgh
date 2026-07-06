"""ROPG-KD: fine-tune the Qwen3-Embedding-0.6B encoder via knowledge distillation from LLM judge scores.

Stage 1 of Simurgh's two-stage training:
  1. Load (query, persona, docs) groups scored by the offline LLM judge
     (notebooks/gen_ropg_data.ipynb, mirroring configs/datagen_ropg.yaml),
     each doc carrying a teacher_score in [0, 1].
  2. Fine-tune the Qwen3-Embedding-0.6B encoder with a LoRA adapter to minimise the listwise
     KL divergence between its similarity distribution over the group's docs and the
     teacher's softmax utility distribution.
  3. Validate every epoch with Recall@K / MRR per persona over the full corpus, and checkpoint
     on the best overall Recall@K.

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
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from personalization.profiles import render_profile

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_groups(path: str | Path) -> list[dict]:
    """Load (query, persona, docs) groups from a ROPG-KD JSONL file.

    Each line is ``{"query": ..., "persona_id": ..., "docs": [{"chunk_id", "text",
    "teacher_score"}, ...]}``. Groups with fewer than 2 docs are dropped (KL over a
    singleton distribution is degenerate).

    Returns:
        List of group dicts, unchanged from the file except for the drop filter.
    """
    groups: list[dict] = []
    n_dropped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if len(rec.get("docs", [])) < 2:
            n_dropped += 1
            continue
        groups.append(rec)
    logger.info(
        "Loaded %d groups from %s (dropped %d with < 2 docs)", len(groups), path, n_dropped
    )
    return groups


def load_corpus(path: str | Path) -> list[dict]:
    """Load the full chunk corpus (fields: chunk_id, text) used for validation retrieval."""
    corpus = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        corpus.append({"chunk_id": rec["chunk_id"], "text": rec["text"]})
    logger.info("Loaded %d corpus chunks from %s", len(corpus), path)
    return corpus


# ── Encoding ──────────────────────────────────────────────────────────────────


def encode_texts(
    st_model, texts: list[str], device: str, instruction: str | None = None
) -> torch.Tensor:
    """Encode *texts* through the same preprocess -> forward path as ``SentenceTransformer.encode``,
    but with gradients enabled so it can be used inside a training loop.

    *instruction*, if given, is rendered as the Qwen3-Embedding instruct prefix
    (``Instruct: {instruction}\\nQuery: ``) and passed as the ``prompt`` kwarg so the
    input module can correctly mask the prompt tokens out of pooling (``prompt_length``).
    """
    import torch.nn.functional as F
    from sentence_transformers.util import batch_to_device

    prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
    features = st_model.preprocess(texts, prompt=prompt)
    features = batch_to_device(features, device)
    out = st_model.forward(features)
    # Cast to fp32 so cosine similarities and the KL loss are computed at full
    # precision even when the forward ran under fp16 autocast (fp16 resolution
    # near 1.0 is too coarse for stable ranking).
    return F.normalize(out["sentence_embedding"], p=2, dim=-1).float()


def encode_docs_chunked(st_model, texts: list[str], device: str, micro_batch: int) -> torch.Tensor:
    """Encode *texts* in micro-batches of *micro_batch* and concatenate.

    Peak activation memory scales with the per-forward batch, so chunking the ~20-doc
    encode keeps one KD step within a 16 GB T4. The concatenated embeddings still sit
    in a single autograd graph, so the listwise loss is unchanged.
    """
    import torch

    vecs = [
        encode_texts(st_model, texts[i : i + micro_batch], device)
        for i in range(0, len(texts), micro_batch)
    ]
    return torch.cat(vecs, dim=0)


# ── Validation ────────────────────────────────────────────────────────────────


def _kd_loss(q_vec: torch.Tensor, d_vecs: torch.Tensor, scores: torch.Tensor, temperature: float):
    """Listwise KD loss: KL(teacher softmax || student softmax) over one group's docs."""
    import torch.nn.functional as F

    sims = (q_vec @ d_vecs.T).squeeze(0)
    targets = F.softmax(scores / temperature, dim=0)
    log_probs = F.log_softmax(sims / temperature, dim=0)
    return F.kl_div(log_probs, targets, reduction="sum")


def evaluate(
    st_model,
    val_groups: list[dict],
    corpus: list[dict],
    device: str,
    kd_temperature: float,
    top_k: int,
    relevance_top_m: int,
    corpus_batch: int = 32,
) -> dict:
    """Run validation: mean KD loss over val groups, plus Recall@K / MRR per persona and overall.

    Ranks the full corpus by cosine similarity to each val group's persona-conditioned query
    embedding; the relevant set for a group is its top-*relevance_top_m* docs by teacher_score
    (matched against the corpus by chunk_id).
    """
    import torch

    st_model.eval()
    # Autocast keeps validation within T4 memory; embeddings are cast back to fp32
    # inside encode_texts before any similarity math.
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        # Re-embed the full corpus with the current model. Group docs *are* corpus
        # chunks, so their embeddings are looked up from this matrix by chunk_id
        # instead of being re-encoded per group (re-encoding ~20 docs x 48 groups
        # made validation ~15x slower than it needs to be).
        from tqdm import tqdm

        corpus_texts = [c["text"] for c in corpus]
        corpus_ids = [c["chunk_id"] for c in corpus]
        corpus_vecs = []
        for i in tqdm(
            range(0, len(corpus_texts), corpus_batch), desc="  eval: corpus", leave=False
        ):
            corpus_vecs.append(encode_texts(st_model, corpus_texts[i : i + corpus_batch], device))
        corpus_matrix = torch.cat(corpus_vecs, dim=0)  # (N, dim)
        chunk_id_to_row = {cid: i for i, cid in enumerate(corpus_ids)}

        # Batch-encode val queries per persona: same instruction means the same
        # prompt, so they can share a forward. Single-query forwards otherwise
        # dominate eval time.
        by_persona: dict[str, list[int]] = {}
        for gi, group in enumerate(val_groups):
            by_persona.setdefault(group["persona_id"], []).append(gi)
        query_vecs: dict[int, torch.Tensor] = {}
        for persona_id, idxs in by_persona.items():
            instruction = render_profile(persona_id)
            for j in range(0, len(idxs), corpus_batch):
                chunk = idxs[j : j + corpus_batch]
                vecs = encode_texts(
                    st_model,
                    [val_groups[gi]["query"] for gi in chunk],
                    device,
                    instruction=instruction,
                )
                for gi, vec in zip(chunk, vecs, strict=True):
                    query_vecs[gi] = vec

        # Val KD loss, computed from the corpus embeddings.
        losses = []
        for gi, group in enumerate(val_groups):
            rows: list[int] = []
            kept_scores: list[float] = []
            for d in group["docs"]:
                row = chunk_id_to_row.get(d["chunk_id"])
                if row is not None:
                    rows.append(row)
                    kept_scores.append(d["teacher_score"])
            if len(rows) < 2:
                continue
            scores = torch.tensor(kept_scores, device=device, dtype=torch.float32)
            q_vec = query_vecs[gi].unsqueeze(0)
            losses.append(_kd_loss(q_vec, corpus_matrix[rows], scores, kd_temperature).item())
        val_loss = float(np.mean(losses)) if losses else float("nan")

        per_persona_recall: dict[str, list[float]] = {}
        per_persona_mrr: dict[str, list[float]] = {}
        all_recall: list[float] = []
        all_mrr: list[float] = []

        for gi, group in enumerate(val_groups):
            persona_id = group["persona_id"]
            relevant_docs = sorted(group["docs"], key=lambda d: d["teacher_score"], reverse=True)[
                :relevance_top_m
            ]
            relevant_rows = {
                chunk_id_to_row[d["chunk_id"]]
                for d in relevant_docs
                if d["chunk_id"] in chunk_id_to_row
            }
            if not relevant_rows:
                continue

            q_vec = query_vecs[gi].unsqueeze(0)
            sims = (q_vec @ corpus_matrix.T).squeeze(0)  # (N,)
            ranked = torch.argsort(sims, descending=True).tolist()

            top_k_rows = set(ranked[:top_k])
            recall = len(relevant_rows & top_k_rows) / len(relevant_rows)

            rr = 0.0
            for rank, row in enumerate(ranked, start=1):
                if row in relevant_rows:
                    rr = 1.0 / rank
                    break

            per_persona_recall.setdefault(persona_id, []).append(recall)
            per_persona_mrr.setdefault(persona_id, []).append(rr)
            all_recall.append(recall)
            all_mrr.append(rr)

    metrics = {
        "val_kd_loss": val_loss,
        "recall_at_k": {"overall": float(np.mean(all_recall)) if all_recall else 0.0},
        "mrr": {"overall": float(np.mean(all_mrr)) if all_mrr else 0.0},
    }
    for persona_id in per_persona_recall:
        metrics["recall_at_k"][persona_id] = float(np.mean(per_persona_recall[persona_id]))
        metrics["mrr"][persona_id] = float(np.mean(per_persona_mrr[persona_id]))

    logger.info(
        "Validation | KD loss: %.4f | Recall@K overall: %.4f | MRR overall: %.4f",
        val_loss,
        metrics["recall_at_k"]["overall"],
        metrics["mrr"]["overall"],
    )
    for persona_id in per_persona_recall:
        logger.info(
            "  persona=%s | Recall@K: %.4f | MRR: %.4f",
            persona_id,
            metrics["recall_at_k"][persona_id],
            metrics["mrr"][persona_id],
        )
    return metrics


def _is_better(candidate: dict, current_best: dict | None) -> bool:
    """Best checkpoint = highest overall Recall@K; ties -> lower val KL; ties -> higher overall MRR."""
    if current_best is None:
        return True
    c_recall = candidate["recall_at_k"]["overall"]
    b_recall = current_best["recall_at_k"]["overall"]
    if c_recall != b_recall:
        return c_recall > b_recall
    if candidate["val_kd_loss"] != current_best["val_kd_loss"]:
        return candidate["val_kd_loss"] < current_best["val_kd_loss"]
    return candidate["mrr"]["overall"] > current_best["mrr"]["overall"]


# ── Training ──────────────────────────────────────────────────────────────────


def train(config: dict) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    data_cfg = config["data"]
    emb_cfg = config["embedder"]
    train_cfg = config["training"]
    lora_cfg_dict = config["lora"]
    eval_cfg = config["eval"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    precision = train_cfg.get("precision", "fp32")
    use_amp = device == "cuda" and precision == "fp16"
    logger.info("Device: %s | precision: %s | amp: %s", device, precision, use_amp)

    # Load data.
    train_groups = load_groups(data_cfg["train_path"])
    val_groups = load_groups(data_cfg["val_path"])
    corpus = load_corpus(data_cfg["corpus_path"])

    # Load the encoder and wrap with a LoRA adapter. Weights stay fp32 even under
    # precision=fp16: GradScaler cannot unscale fp16 gradients, so fp16 applies to
    # the autocast forward pass only (the standard AMP recipe).
    model_name = emb_cfg.get("model", "Qwen/Qwen3-Embedding-0.6B")
    attn_impl = emb_cfg.get("attn_implementation", "sdpa")
    st_model = SentenceTransformer(
        model_name, device=device, trust_remote_code=True,
        model_kwargs={"attn_implementation": attn_impl},
    )
    # get_peft_model injects the LoRA layers and freezes the base weights *in place*
    # on the passed model, so SentenceTransformer's forward path trains through them
    # without reassignment (Transformer.auto_model is a read-only property; assigning
    # to it would be silently shadowed by nn.Module.__setattr__). The wrapper is kept
    # only for print_trainable_parameters() and adapter-only save_pretrained().
    peft_model = get_peft_model(
        st_model[0].auto_model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_cfg_dict.get("r", 8),
            lora_alpha=lora_cfg_dict.get("alpha", 16),
            lora_dropout=lora_cfg_dict.get("dropout", 0.1),
            target_modules=lora_cfg_dict.get("target_modules", ["q_proj", "v_proj"]),
        ),
    )
    peft_model.print_trainable_parameters()

    # One KD step keeps the activation graphs of a query + ~20 docs alive until the
    # backward pass, which OOMs a 16 GB T4 without checkpointing. use_reentrant=False
    # lets gradients flow even though only the LoRA params require grad.
    # set gradient_checkpointing: true in config if OOM
    if train_cfg.get("gradient_checkpointing", False):
        st_model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Cap tokenization length: the tokenizer default (32k) lets one long outlier
    # blow up the whole padded batch; real chunks are far shorter than 8k tokens.
    st_model[0].max_seq_length = emb_cfg.get("max_seq_length", 8192)

    # Diagnostics: confirm checkpointing actually engaged and how big batches really
    # are in tokens — the evidence needed to aim any further OOM fix.
    auto_model = st_model[0].auto_model
    logger.info(
        "Gradient checkpointing active: %s | attn implementation: %s",
        getattr(auto_model, "is_gradient_checkpointing", "unknown"),
        getattr(auto_model.config, "_attn_implementation", "unknown"),
    )
    tokenizer = st_model[0].tokenizer
    if tokenizer is not None:
        lengths = sorted(len(tokenizer(c["text"])["input_ids"]) for c in corpus)
        logger.info(
            "Corpus token lengths | max: %d | p95: %d | median: %d",
            lengths[-1],
            lengths[int(0.95 * (len(lengths) - 1))],
            lengths[len(lengths) // 2],
        )

    trainable_params = [p for p in st_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.get("lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    n_epochs = train_cfg.get("epochs", 5)
    batch_size = train_cfg.get("batch_size", 8)  # groups per optimizer step
    kd_temp = train_cfg.get("kd_temperature", 1.0)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)

    n_steps_per_epoch = max(1, math.ceil(len(train_groups) / batch_size))
    total_steps = n_steps_per_epoch * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    top_k = eval_cfg.get("top_k", 5)
    relevance_top_m = eval_cfg.get("relevance_top_m", 3)
    doc_micro_batch = train_cfg.get("doc_micro_batch", 8)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Epoch-0 baseline: validate the frozen (untrained) encoder before any training step.
    logger.info("Running epoch-0 baseline validation (frozen encoder)...")
    baseline_metrics = evaluate(
        st_model, val_groups, corpus, device, kd_temp, top_k, relevance_top_m, doc_micro_batch
    )

    epoch_metrics: list[dict] = []
    best_metrics: dict | None = None
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_groups)
        st_model.train()
        epoch_loss = 0.0
        n_groups_seen = 0

        optimizer.zero_grad()
        for i, group in enumerate(tqdm(train_groups, desc=f"Epoch {epoch}/{n_epochs}")):
            docs = sorted(group["docs"], key=lambda d: d["teacher_score"], reverse=True)[:8]
            scores = torch.tensor(
                [d["teacher_score"] for d in docs], device=device, dtype=torch.float32
            )
            doc_texts = [d["text"] for d in docs]
            instruction = render_profile(group["persona_id"])

            if use_amp:
                with torch.amp.autocast("cuda"):
                    q_vec = encode_texts(
                        st_model, [group["query"]], device, instruction=instruction
                    )
                    d_vecs = encode_docs_chunked(st_model, doc_texts, device, doc_micro_batch)
                    loss = _kd_loss(q_vec, d_vecs, scores, kd_temp) / batch_size
                scaler.scale(loss).backward()
            else:
                q_vec = encode_texts(st_model, [group["query"]], device, instruction=instruction)
                d_vecs = encode_docs_chunked(st_model, doc_texts, device, doc_micro_batch)
                loss = _kd_loss(q_vec, d_vecs, scores, kd_temp) / batch_size
                loss.backward()

            epoch_loss += loss.item() * batch_size
            n_groups_seen += 1

            if n_groups_seen % batch_size == 0 or i == len(train_groups) - 1:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = epoch_loss / max(n_groups_seen, 1)
        logger.info("Epoch %d/%d | avg train KL loss: %.4f", epoch, n_epochs, avg_train_loss)

        if device == "cuda":
            logger.info(
                "CUDA peak memory | allocated: %.2f GB | reserved: %.2f GB",
                torch.cuda.max_memory_allocated() / 1e9,
                torch.cuda.max_memory_reserved() / 1e9,
            )
            torch.cuda.empty_cache()
        val_metrics = evaluate(
            st_model, val_groups, corpus, device, kd_temp, top_k, relevance_top_m, doc_micro_batch
        )
        val_metrics["epoch"] = epoch
        val_metrics["train_kd_loss"] = avg_train_loss
        epoch_metrics.append(val_metrics)

        epoch_dir = checkpoint_dir / f"epoch{epoch}"
        peft_model.save_pretrained(str(epoch_dir))
        (epoch_dir / "metrics.json").write_text(
            json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Checkpoint saved: %s", epoch_dir)

        if _is_better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch

    best_dir = checkpoint_dir / "best"
    if best_epoch > 0:
        source_dir = checkpoint_dir / f"epoch{best_epoch}"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(source_dir, best_dir)
        logger.info("Best checkpoint (epoch %d) copied to %s", best_epoch, best_dir)

    training_log = {
        "config": config,
        "seed": seed,
        "baseline_metrics": baseline_metrics,
        "epoch_metrics": epoch_metrics,
        "best_epoch": best_epoch,
        "best_epoch_reason": (
            "highest overall Recall@K; ties broken by lower val KL loss, then higher overall MRR"
        ),
    }
    (checkpoint_dir / "training_log.json").write_text(
        json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Training log written: %s", checkpoint_dir / "training_log.json")


def main() -> None:
    # Must be set before torch initializes CUDA; mitigates fragmentation OOMs by
    # letting the allocator grow segments instead of hunting for contiguous blocks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="ROPG-KD: fine-tune Qwen3-Embedding encoder with KD."
    )
    parser.add_argument("--config", required=True, help="Path to train_ropg.yaml.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
