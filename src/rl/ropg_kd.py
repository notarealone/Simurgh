"""ROPG-KD: fine-tune the Qwen3-Embedding-0.6B encoder via knowledge distillation from LLM judge scores.

Stage 1 of Simurgh's two-stage training:
  1. Load (query, persona, docs) groups scored by the offline LLM judge
     (notebooks/gen_ropg_data.ipynb, mirroring configs/datagen_ropg.yaml),
     each doc carrying a teacher_score in [0, 1].
  2. Fine-tune a LoRA adapter over Qwen3-Embedding-0.6B to minimise the listwise
     KL divergence between its similarity distribution over the group's docs and the
     teacher's softmax utility distribution.
  3. Validate every epoch with Recall@K / MRR **per persona over the full corpus**
     and checkpoint on Recall@K (ties: lower val KL, then higher MRR).

Supports two training modes:
  - hard_neg: Multiple Negatives Ranking Loss (MNRL) over rank-1 vs. tail — **primary**
  - reader_kd: listwise KL-distillation against the teacher's graded scores — retained
    as a documented ablation, not the primary arm. Its teacher is a single-sample
    LLM-judge rating (per-label reliability ~0.17), and KL weights the whole graded
    ranking including the middle, where adjacent teacher gaps sit below that noise
    floor. See docs/methodology.md, "Why reader_kd was retired".

``hard_neg`` consumes pre-derived triplets and never builds them: derivation is owned by
``python -m data.gen_ropg_data --config configs/datagen_ropg.yaml --derive-only``, which is
free (no API calls, no model load). Deriving here would have to guess the label filters,
which live in the datagen config, so a missing file would quietly yield an unfiltered run.
The trainer instead validates ``train_triplets_meta.json`` against
``training.max_negatives`` and refuses to start on stale data.

The encoder here MUST stay byte-for-byte equivalent to inference
(``rag.embedder.Qwen3Embedder``): last-token pooling, documents encoded bare, and
queries wrapped as ``Instruct: {persona}\\nQuery: {text}``. ``benchmarks/check_encoder_parity.py``
is the gate on that and asserts cosine >= 0.999 on both sides; a pooling or prompt mismatch
silently destroys the fine-tune while the loss curve still looks healthy.

Runs on GPU (Kaggle / university cluster). Install the training extras first:
    uv sync --extra embedding --extra training

Multi-GPU is launched exclusively with ``torchrun``; there is no in-process
spawning. ``main_worker`` reads ``LOCAL_RANK``/``RANK``/``WORLD_SIZE`` from the
environment, which torchrun sets and which default to a single process otherwise.

Typical usage:
    uv run python -m rl.ropg_kd --config configs/train_ropg.yaml
    torchrun --standalone --nproc_per_node=2 -m rl.ropg_kd --config configs/train_ropg.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from personalization.profiles import render_profile, train_personas

logger = logging.getLogger(__name__)


# =============================================================================
# Inference-parity helpers
# =============================================================================


def format_query(query: str, persona_id: str | None) -> str:
    """Render a query exactly as ``Qwen3Embedder.encode_query`` does at inference.

    ``rag.embedder.Qwen3Embedder`` passes ``prompt=f"Instruct: {instruction}\\nQuery: "``
    to sentence-transformers, which concatenates it in front of the text. Reproducing
    that string here is what keeps the fine-tuned adapter usable by the serving path.
    Documents are deliberately *not* wrapped — inference encodes them bare.
    """
    if not persona_id:
        return query
    return f"Instruct: {render_profile(persona_id)}\nQuery: {query}"


# Persona-swap control (see ``evaluate_retrieval``). A rotation over the *train*
# personas, in their declared order, so every group is re-rendered under a persona that
# is not its own while the per-persona cell sizes stay identical. Derived rather than
# hard-coded: adding a fourth training persona extends the cycle automatically, and the
# held-out test persona is deliberately excluded — introducing it here would leak the
# holdout into a validation-time diagnostic.
_TRAIN_PERSONA_IDS = [p.id for p in train_personas()]
PERSONA_ROTATION: dict[str, str] = {
    pid: _TRAIN_PERSONA_IDS[(i + 1) % len(_TRAIN_PERSONA_IDS)]
    for i, pid in enumerate(_TRAIN_PERSONA_IDS)
}


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Qwen3-Embedding's native pooling: take the final non-padding token.

    Padding-side agnostic, matching the reference implementation on the model card.
    sentence-transformers applies the same pooling at inference, so this is the
    single most important line for train/serve consistency.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        seq_lens,
    ]


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module whether or not *model* is DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model


# =============================================================================
# Datasets — persona-conditioned, chunk_id-preserving
# =============================================================================


class TripletDataset(Dataset):
    """Reads {query, persona_id, positive, negatives: [...]} lines (hard_neg mode)."""

    def __init__(self, jsonl_path: str, max_negatives: int = 4) -> None:
        self.items: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                negs = obj.get("negatives", [])[:max_negatives]
                if not negs:
                    continue
                self.items.append(
                    {
                        "query": format_query(obj["query"], obj.get("persona_id")),
                        "positive": obj["positive"],
                        "negatives": negs,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class PairDataset(Dataset):
    """Reads {query, persona_id, positive, negative} lines (cartesian-expanded)."""

    def __init__(self, jsonl_path: str) -> None:
        self.items: list[dict[str, str]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.items.append(
                    {
                        "query": format_query(obj["query"], obj.get("persona_id")),
                        "positive": obj["positive"],
                        "negative": obj["negative"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.items[idx]


class ScoredDataset(Dataset):
    """Reads listwise scored groups for KL-distillation.

    Supports two JSONL formats:
      1. Old: {"query": "...", "document": "...", "score": 0.5}
      2. New: {"query": "...", "persona_id": "...",
               "docs": [{"chunk_id": "...", "text": "...", "teacher_score": 0.3}, ...]}

    Groups are keyed on ``(raw query, persona_id)``. Keying on the *rendered* query
    alone would merge the same question asked of different personas — the exact
    signal this stage is trying to learn. ``chunk_id`` and ``persona_id`` are kept
    on each item because corpus-level per-persona evaluation needs both.
    """

    def __init__(self, jsonl_path: str, max_documents: int = 20) -> None:
        grouped: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
        order: list[tuple[str, str]] = []
        n_dup = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                persona_id = obj.get("persona_id", "")
                key = (obj["query"], persona_id)
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                else:
                    n_dup += 1

                if "docs" in obj:
                    for doc in obj["docs"]:
                        grouped[key].append(
                            (
                                doc["text"],
                                float(doc["teacher_score"]),
                                doc.get("chunk_id", ""),
                            )
                        )
                elif "document" in obj and "score" in obj:
                    grouped[key].append(
                        (obj["document"], float(obj["score"]), obj.get("chunk_id", ""))
                    )
                else:
                    continue  # unknown format

        if n_dup:
            logger.warning(
                "%s: %d duplicate (query, persona_id) rows merged into existing groups; "
                "their docs beyond max_documents=%d are dropped.",
                Path(jsonl_path).name,
                n_dup,
                max_documents,
            )

        self.items: list[dict[str, Any]] = []
        for key in order:
            triples = grouped[key][:max_documents]
            if len(triples) < 2:
                continue
            docs, scores, chunk_ids = zip(*triples, strict=True)
            raw_query, persona_id = key
            self.items.append(
                {
                    "query": format_query(raw_query, persona_id),
                    "raw_query": raw_query,
                    "persona_id": persona_id,
                    "documents": list(docs),
                    "scores": list(scores),
                    "chunk_ids": list(chunk_ids),
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
    """Qwen3-Embedding wrapper: last-token pooling + L2 normalization, optional LoRA."""

    def __init__(
        self,
        model_name: str,
        use_gradient_checkpointing: bool = True,
        model_dtype: torch.dtype = torch.float16,
        lora_config: dict | None = None,
    ) -> None:
        super().__init__()
        # Eager attention materializes the full attention score matrix *and* upcasts it
        # for the softmax, costing rows x heads x width^2 x 4 bytes - 2.5 GB for a single
        # 10-sequence micro-batch at width 2048, allocated again during gradient
        # checkpointing's backward recompute. That is what OOM'd a 2xT4 run mid-epoch.
        # SDPA either drops the matrix entirely (mem-efficient kernel; flash needs sm_80,
        # so not on a T4) or falls back to math, which still materializes it but in fp16
        # with no fp32 softmax - half the memory. Both outcomes fit; there is no reason to
        # prefer eager here. No try/except: if this checkpoint cannot do SDPA we want the
        # ValueError now, not an OOM hours in with nothing in the log to explain it.
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        )
        # Caching is incompatible with gradient checkpointing and useless for encoding.
        self.model.config.use_cache = False
        # Log what was actually installed - transformers can quietly resolve to something
        # other than the request, and the difference is a 2x swing in peak memory.
        logger.info(
            "attn_implementation=%s", getattr(self.model.config, "_attn_implementation", "?")
        )

        if lora_config is not None:
            from peft import LoraConfig, TaskType, get_peft_model

            # With a frozen base, the input to the first checkpointed block has
            # requires_grad=False, so recomputation yields no gradient at all.
            # This hook is what makes gradient checkpointing + PEFT actually train.
            if use_gradient_checkpointing:
                self.model.enable_input_require_grads()
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("alpha", 16),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=list(
                    lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
                ),
            )
            self.model = get_peft_model(self.model, peft_cfg)

        if use_gradient_checkpointing:
            # use_reentrant=False is required for DDP: the reentrant autograd path
            # hides parameter usage from DDP's bucketing and trips "marked ready twice".
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(outputs.last_hidden_state, attention_mask)
        # fp32 before normalising: fp16 resolution near 1.0 is too coarse to rank
        # cosine similarities stably, and every downstream loss is a softmax over them.
        return F.normalize(emb.float(), p=2, dim=1)

    @contextmanager
    def base_only(self):
        """Encode with the LoRA adapter switched off — the pretrained model's own output.

        This is what makes anchoring cheap: the reference embedding comes from the same
        weights already in memory, so no second copy of the model is needed. A no-op when
        training without LoRA, where the base and the trained model are the same thing.
        """
        disable = getattr(self.model, "disable_adapter", None)
        if disable is None:
            yield
        else:
            with disable():
                yield

    def save_adapter(self, path: str | Path) -> None:
        """Persist LoRA weights (or the full model when training without LoRA)."""
        self.model.save_pretrained(str(path))


def trim_pad(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop leading columns that are padding in *every* row of this micro-batch.

    Padding is left-side (``tokenizer.padding_side = "left"``), so real tokens are
    right-aligned and slicing ``[:, -keep:]`` can only remove pad. Nothing the model
    would have used is lost: pad positions are already excluded by the attention mask,
    and ``last_token_pool`` reads the final token, which left padding never touches.
    The absolute position of every real token shifts by the same amount, which is
    invisible to RoPE — it encodes *relative* position.
    """
    keep = int(attention_mask.sum(dim=1).max().item())
    return input_ids[:, -keep:], attention_mask[:, -keep:]


def encode_chunked(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    micro_batch: int,
) -> torch.Tensor:
    """Encode in length-sorted micro-batches and concatenate, keeping one autograd graph.

    The concatenated result is a single graph, so the listwise loss and its gradients
    are identical to one large forward. That also bounds what chunking buys on memory:
    because the graph is retained until ``backward()``, *stored* activations scale with
    total tokens per step regardless of *micro_batch* — ``batch_size`` is the knob for
    those. What *micro_batch* caps is the transient per-forward working set.

    It is also a padding lever, and that is where the real cost sits. ``collate_scored``
    pads to the longest sequence in the whole batch, but each micro-batch is its own
    forward and needs only its own width. Sorting by true length concentrates the few
    long documents into one micro-batch instead of letting each drag its neighbours up
    to 2048; ``trim_pad`` then slices off the columns that are pad for the entire chunk.
    On this corpus (median document 167 tokens, p95 2048) that removes ~65% of all
    tokens processed. Rows are restored to the caller's order before returning, which
    both call sites depend on — they ``view(B, D, -1)`` the result against
    ``gold_scores``.

    Consequence for tuning: a smaller *micro_batch* now buys real compute, not just a
    smaller transient buffer, traded against starving the GPU on tiny launches.
    """
    n = input_ids.size(0)
    if micro_batch <= 0 or n <= micro_batch:
        return model(*trim_pad(input_ids, attention_mask))

    # stable=True keeps the permutation reproducible under the run seed.
    order = torch.argsort(attention_mask.sum(dim=1), stable=True)
    parts = [
        model(*trim_pad(input_ids[sel], attention_mask[sel]))
        for sel in (order[i : i + micro_batch] for i in range(0, n, micro_batch))
    ]
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n, device=order.device)
    return torch.cat(parts, dim=0)[inverse]


# =============================================================================
# Losses
# =============================================================================


def mnrl_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.05,
    neg_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiple Negatives Ranking Loss over one positive and N in-row negatives.

    *neg_mask* removes padded negative slots from the softmax. ``collate_triplets``
    pads short rows with empty strings so the batch is rectangular, and an empty
    string still encodes to a real vector — left unmasked it competes with the
    positive as though it were a genuine hard negative. This was latent while every
    group carried exactly ``max_negatives`` negatives; ``min_negative_margin`` in
    ``derive_triplets`` makes row lengths ragged, which is what activates it.
    """
    q = query_emb.float()
    pos_score = (q * pos_emb.float()).sum(dim=-1, keepdim=True)
    neg_scores = torch.bmm(neg_emb.float(), q.unsqueeze(-1)).squeeze(-1)
    if neg_mask is not None:
        neg_scores = neg_scores.masked_fill(~neg_mask, float("-inf"))
    all_scores = torch.cat([pos_score, neg_scores], dim=-1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(all_scores, labels)


def kd_loss(
    student_scores: torch.Tensor,
    gold_scores: torch.Tensor,
    student_temp: float = 0.05,
    teacher_temp: float = 0.2,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Listwise KL(teacher || student) over each group's candidate documents.

    Both temperatures matter more than they look. Student scores are cosine
    similarities in [-1, 1] and teacher scores lie in [0, 1]; at temperature 1.0 a
    softmax over 20 candidates is nearly uniform on *both* sides, leaving almost no
    gradient. *doc_mask* removes padded slots from the student's normalisation so
    they cannot absorb probability mass.
    """
    student = student_scores.float() / student_temp
    teacher = gold_scores.float() / teacher_temp
    if doc_mask is not None:
        student = student.masked_fill(~doc_mask, float("-inf"))
        teacher = teacher.masked_fill(~doc_mask, float("-inf"))
    student_logprobs = F.log_softmax(student, dim=-1)
    gold_probs = F.softmax(teacher, dim=-1)
    if doc_mask is not None:
        # log_softmax leaves -inf in the padded slots, and kl_div evaluates
        # target * (log target - input) there as 0 * inf = NaN. The teacher weight is
        # already 0 on pads, so any finite value works; zero keeps the term at exactly 0.
        student_logprobs = student_logprobs.masked_fill(~doc_mask, 0.0)
    return F.kl_div(student_logprobs, gold_probs, reduction="batchmean")


# =============================================================================
# Collators
# =============================================================================


def collate_triplets(
    batch: list[dict[str, Any]], tokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    queries = [x["query"] for x in batch]
    positives = [x["positive"] for x in batch]
    max_negs = max(len(x["negatives"]) for x in batch)

    negs_flat: list[str] = []
    neg_mask: list[list[bool]] = []
    for x in batch:
        n = len(x["negatives"])
        negs_flat.extend(x["negatives"])
        negs_flat.extend([""] * (max_negs - n))
        neg_mask.append([True] * n + [False] * (max_negs - n))

    def tok(texts: list[str]):
        return tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    q, p, n_tok = tok(queries), tok(positives), tok(negs_flat)
    B = len(batch)
    out = {
        "q_ids": q["input_ids"],
        "q_mask": q["attention_mask"],
        "p_ids": p["input_ids"],
        "p_mask": p["attention_mask"],
        "n_ids": n_tok["input_ids"].view(B, max_negs, -1),
        "n_mask": n_tok["attention_mask"].view(B, max_negs, -1),
        "neg_mask": torch.tensor(neg_mask, dtype=torch.bool),
    }
    # Row ids into the anchor cache, present only once build_anchor_cache has annotated
    # the dataset. Padded negative slots point at row 0; neg_mask is what excludes them.
    if "q_base" in batch[0]:
        out["q_base"] = torch.tensor([x["q_base"] for x in batch], dtype=torch.long)
        out["p_base"] = torch.tensor([x["p_base"] for x in batch], dtype=torch.long)
        out["n_base"] = torch.tensor(
            [x["n_base"] + [0] * (max_negs - len(x["n_base"])) for x in batch], dtype=torch.long
        )
    return out


def collate_pairs(
    batch: list[dict[str, str]], tokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    triplet_batch = [
        {"query": x["query"], "positive": x["positive"], "negatives": [x["negative"]]}
        for x in batch
    ]
    return collate_triplets(triplet_batch, tokenizer, max_length)


def collate_scored(
    batch: list[dict[str, Any]], tokenizer, max_length: int
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
        gold_scores.append(list(x["scores"]) + [0.0] * (max_docs - n))
        doc_mask.append([True] * n + [False] * (max_docs - n))

    def tok(texts: list[str]):
        return tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    q, d = tok(queries), tok(docs_flat)
    B = len(batch)
    return {
        "q_ids": q["input_ids"],
        "q_mask": q["attention_mask"],
        "d_ids": d["input_ids"].view(B, max_docs, -1),
        "d_mask": d["attention_mask"].view(B, max_docs, -1),
        "gold_scores": torch.tensor(gold_scores, dtype=torch.float32),
        "doc_mask": torch.tensor(doc_mask, dtype=torch.bool),
    }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# =============================================================================
# Train / eval steps
# =============================================================================


def _kd_step(model, batch, temps, micro_batch):
    student_temp, teacher_temp = temps
    q_ids, q_mask = batch["q_ids"], batch["q_mask"]
    d_ids, d_mask = batch["d_ids"], batch["d_mask"]
    B, D, L = d_ids.shape
    q_emb = encode_chunked(model, q_ids, q_mask, micro_batch)
    d_emb = encode_chunked(model, d_ids.view(B * D, L), d_mask.view(B * D, L), micro_batch).view(
        B, D, -1
    )
    student_scores = torch.einsum("bh,bdh->bd", q_emb, d_emb)
    return kd_loss(
        student_scores,
        batch["gold_scores"],
        student_temp,
        teacher_temp,
        batch["doc_mask"],
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    step_fn,
    temps,
    amp_dtype,
    use_amp,
    grad_clip,
    micro_batch,
    is_main: bool = True,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False, disable=not is_main)
    for batch in pbar:
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            loss = step_fn(model, batch, temps, micro_batch)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # GradScaler starts at 65536, so the first fp16 backward usually overflows
            # and scaler.step() *skips* the update, halving the scale. Advancing the LR
            # schedule for an update that never happened is what triggers PyTorch's
            # "lr_scheduler.step() before optimizer.step()" warning - the call order
            # here is already the documented one; the skip is what it is complaining
            # about. A lowered scale is the signal that the step was dropped.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            stepped = True
        if stepped:
            scheduler.step()

        total += loss.item()
        n += 1
        if is_main:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, step_fn, temps, amp_dtype, use_amp, micro_batch) -> float:
    """Mean loss over the loader, all-reduced so every rank reports the *whole* val set.

    Without the reduction a DistributedSampler-sharded loader makes each rank report
    only its own slice, which then silently drives checkpoint selection.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="eval", leave=False, disable=True):
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            loss = step_fn(model, batch, temps, micro_batch)
        total += loss.item()
        n += 1

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([total, float(n)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = stats[0].item(), int(stats[1].item())
    return total / max(n, 1)


# =============================================================================
# Corpus-level evaluation (persona-aware)
# =============================================================================


def load_corpus(corpus_path: str) -> tuple[list[str], dict[str, int]]:
    """Load data/chunks/corpus.jsonl → (texts, chunk_id → row index)."""
    texts: list[str] = []
    chunk_id_to_row: dict[str, int] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunk_id_to_row[obj["chunk_id"]] = len(texts)
            texts.append(obj["text"])
    return texts, chunk_id_to_row


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    amp_dtype: torch.dtype = torch.float16,
    use_amp: bool = True,
    desc: str | None = None,
) -> torch.Tensor:
    model.eval()
    out: list[torch.Tensor] = []
    rng = range(0, len(texts), batch_size)
    for i in tqdm(rng, desc=desc, leave=False, disable=desc is None):
        tokens = tokenizer(
            texts[i : i + batch_size],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            emb = model(tokens["input_ids"], tokens["attention_mask"])
        out.append(emb.float())
    return torch.cat(out, dim=0)


# =============================================================================
# Base-model anchoring
# =============================================================================


@torch.no_grad()
def build_anchor_cache(
    model,
    tokenizer,
    datasets: list[Dataset],
    device: torch.device,
    batch_size: int,
    max_length: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> torch.Tensor:
    """Encode every distinct text across *datasets* with the adapter off, once, up front.

    Returns the cache matrix and annotates each item in place with ``q_base``/``p_base``/
    ``n_base`` row ids into it, which ``collate_triplets`` forwards to the loss. Train and
    val share one cache and one row map: the val loader runs the same step function, so a
    val item with no row id would fault on the anchor lookup.

    Caching rather than recomputing is what makes anchoring free. The alternative — a
    second ``base_only`` forward inside every step — is simpler but costs one extra
    forward per step, roughly +40% wall clock. It is affordable here only because the
    corpus is 171 chunks: unique texts run to ~1.5k, so the cache is a few MB and the
    encode is a one-off of a couple of minutes against multi-hour epochs.
    """
    texts: list[str] = []
    row_of: dict[str, int] = {}

    def row(text: str) -> int:
        if text not in row_of:
            row_of[text] = len(texts)
            texts.append(text)
        return row_of[text]

    for dataset in datasets:
        for item in dataset.items:
            item["q_base"] = row(item["query"])
            item["p_base"] = row(item["positive"])
            item["n_base"] = [row(t) for t in item["negatives"]]

    with unwrap(model).base_only():
        cache = encode_texts(
            model,
            tokenizer,
            texts,
            device,
            batch_size,
            max_length,
            amp_dtype,
            use_amp,
            desc="anchor: base embeddings",
        )
    logger.info("Anchor cache: %d distinct texts, %.1f MB", len(texts), cache.numel() * 4 / 1e6)
    return cache


def anchor_penalty(emb: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance from the pretrained embedding. Zero when nothing has moved."""
    return (1.0 - (emb.float() * base.float()).sum(dim=-1)).mean()


def make_mnrl_step(anchor: dict | None, cache: torch.Tensor | None):
    """Build the hard_neg step function, closing over the anchor config and cache.

    A closure rather than extra parameters because ``train_epoch``/``eval_epoch`` call
    ``step_fn(model, batch, temps, micro_batch)`` generically for both training modes;
    widening that signature would touch the reader_kd path for no reason.

    Anchoring exists because the failure mode here is *drift*, not underfitting: every
    trained checkpoint so far scores at or below the untrained baseline while train loss
    collapses. Two lambdas rather than one, because personalisation is a property of the
    query side only — documents carry no persona — so the document tower should barely
    move while the query tower is where persona conditioning has to live.
    """
    mode = (anchor or {}).get("mode", "none")
    lambda_doc = float((anchor or {}).get("lambda_doc", 0.0))
    lambda_query = float((anchor or {}).get("lambda_query", 0.0))

    def step(model, batch, temps, micro_batch):
        student_temp, _ = temps
        q_ids, q_mask = batch["q_ids"], batch["q_mask"]
        p_ids, p_mask = batch["p_ids"], batch["p_mask"]
        n_ids, n_mask = batch["n_ids"], batch["n_mask"]
        B, N, L = n_ids.shape

        q_emb = encode_chunked(model, q_ids, q_mask, micro_batch)
        if mode == "doc_frozen":
            # The document tower is the pretrained model outright: no adapter, no grad.
            # Strictly stronger than penalising doc drift, and it halves the trainable
            # path. The serving index must then be built with the adapter off too — see
            # the `anchor.mode` comment in configs/train_ropg.yaml.
            with torch.no_grad(), unwrap(model).base_only():
                p_emb = encode_chunked(model, p_ids, p_mask, micro_batch)
                n_emb = encode_chunked(
                    model, n_ids.view(B * N, L), n_mask.view(B * N, L), micro_batch
                ).view(B, N, -1)
        else:
            p_emb = encode_chunked(model, p_ids, p_mask, micro_batch)
            n_emb = encode_chunked(
                model, n_ids.view(B * N, L), n_mask.view(B * N, L), micro_batch
            ).view(B, N, -1)

        loss = mnrl_loss(q_emb, p_emb, n_emb, student_temp, batch.get("neg_mask"))

        if mode == "none" or cache is None:
            return loss
        if lambda_query:
            loss = loss + lambda_query * anchor_penalty(q_emb, cache[batch["q_base"]])
        # In doc_frozen the documents *are* the base model, so a doc penalty would be
        # identically zero and carry no gradient — skip it rather than pay the lookup.
        if lambda_doc and mode != "doc_frozen":
            doc_emb = torch.cat([p_emb, n_emb[batch["neg_mask"]]], dim=0)
            doc_base = torch.cat(
                [cache[batch["p_base"]], cache[batch["n_base"][batch["neg_mask"]]]]
            )
            loss = loss + lambda_doc * anchor_penalty(doc_emb, doc_base)
        return loss

    return step


@torch.no_grad()
def evaluate_retrieval(
    model,
    tokenizer,
    groups: list[dict[str, Any]],
    corpus_texts: list[str],
    chunk_id_to_row: dict[str, int],
    device: torch.device,
    top_k: int,
    relevance_top_m: int,
    batch_size: int = 32,
    max_length: int = 512,
    amp_dtype: torch.dtype = torch.float16,
    use_amp: bool = True,
    doc_base_only: bool = False,
    swap_personas: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """nDCG@1..K, Hit@1..K, Recall@K, MRR and judged@K against the **full corpus**.

    *doc_base_only* encodes the corpus with the adapter disabled, which ``anchor.mode:
    doc_frozen`` requires: that arm trains the query tower against a frozen document
    tower, so scoring it against an adapted index would measure a pairing that never
    exists at train time and will not exist at serve time either.

    Returns ``(metrics, per_query)``: aggregates per persona and overall, plus the
    raw per-group vectors behind them. The vectors are what make significance testing
    possible — a mean can yield a marginal confidence interval but never a *paired*
    one, and the paired test is the powerful one here because every epoch scores the
    same queries. They are aligned by position, which is sound because group order and
    the skip decision below depend only on the data, never on the model.

    Ranking a group's own 20 candidates measures reranking accuracy, not retrieval;
    the real question is whether the relevant chunks surface out of the whole index.

    Two relevance notions are reported deliberately:

    * **nDCG** (primary) grades by raw ``teacher_score``. Gain is linear, not
      ``2^rel - 1``: scores already live in [0, 1], so exponential gain only compresses
      them. Grading matters because the teacher's scores are far from flat — mean score
      by rank runs 0.757 / 0.569 / 0.447 / 0.372 / 0.323 — and because any binary cutoff
      lands on a near-tie: the rank-3 to rank-4 gap is under 0.05 in 53% of val groups.
      nDCG also has ceiling 1.0 at *every* k, so nDCG@1 is directly readable.
    * **Recall/Hit/MRR** keep the binary top-*relevance_top_m* set. Retained on purpose:
      nDCG is shaped like the KD objective (both range over the teacher's graded
      distribution), so a coarser, differently-shaped metric belongs beside it.

    * **judged@K** is a diagnostic, not a quality score: the fraction of the returned
      top-K that the teacher actually judged. Only a group's own ~20 judged chunks
      carry gain, so a model that surfaces *unjudged but relevant* chunks is punished
      by nDCG for it. Recall rising while nDCG falls fits both "the graded middle got
      worse" and "the retrieved set moved outside the judged pool"; judged@K is what
      separates them. Read it beside nDCG, never instead of it.

    Note Recall@k divides by the size of that set, so Recall@1 could never exceed
    1/*relevance_top_m*; it is reported at k=K only, where the ceiling is 1.0.

    *swap_personas* additionally scores every query under a **rotated** persona
    (crammer→scholar→steady→crammer) against the same corpus embedding, and returns
    the result under ``metrics["persona_swap"]``. This is the control for the thesis
    claim: personas are only a prompt prefix, so an encoder that ignores that prefix
    still improves every headline metric while personalising nothing. Measured on the
    val labels, ranking with the wrong persona costs 0.33 nDCG@5, so a genuinely
    persona-sensitive encoder must degrade visibly here. Rotation is used rather than a
    random reassignment because it keeps the 92/92/92 persona balance exact and is
    deterministic across runs.

    Only the group's own judged chunks have gains — the other ~150 corpus chunks score
    0 even if genuinely relevant. That incomplete-judgments bias predates nDCG and
    applies to Recall/MRR identically; see docs/methodology.md.

    The corpus is embedded once and reused across groups — re-encoding each group's
    docs made validation roughly 15x slower for identical numbers.
    """
    corpus_ctx = model.base_only() if doc_base_only else nullcontext()
    with corpus_ctx:
        corpus_matrix = encode_texts(
            model,
            tokenizer,
            corpus_texts,
            device,
            batch_size,
            max_length,
            amp_dtype,
            use_amp,
            desc="eval: corpus",
        )

    # Position i (0-based) in a ranking carries discount 1/log2(i+2).
    discount = 1.0 / np.log2(np.arange(2, top_k + 2))
    ks = list(range(1, top_k + 1))
    names = (
        [f"ndcg@{k}" for k in ks]
        + [f"hit@{k}" for k in ks]
        + [f"recall@{top_k}", "mrr", f"judged@{top_k}"]
    )

    def score(query_texts: list[str], desc: str | None, warn: bool) -> tuple[dict, dict]:
        """Rank *query_texts* against the shared corpus embedding and aggregate.

        Factored out so the persona-swap pass reuses ``corpus_matrix`` — re-encoding
        171 chunks a second time would double the cost of the whole eval for nothing.
        """
        query_matrix = encode_texts(
            model,
            tokenizer,
            query_texts,
            device,
            batch_size,
            max_length,
            amp_dtype,
            use_amp,
            desc=desc,
        )
        sims_all = query_matrix @ corpus_matrix.T  # (G, N)

        values: dict[str, list[float]] = {name: [] for name in names}
        personas: list[str] = []
        n_skipped = 0

        for gi, group in enumerate(groups):
            scored = sorted(
                zip(group["chunk_ids"], group["scores"], strict=True),
                key=lambda t: t[1],
                reverse=True,
            )
            # Gains only for judged chunks that exist in the corpus. Restricting the
            # ideal ranking the same way keeps nDCG's ceiling attainable — grading
            # against chunks the retriever cannot return would depress every score by
            # a constant.
            gains_by_row = {
                chunk_id_to_row[cid]: float(s) for cid, s in scored if cid in chunk_id_to_row
            }
            relevant_rows = {
                chunk_id_to_row[cid]
                for cid, _ in scored[:relevance_top_m]
                if cid in chunk_id_to_row
            }
            if not relevant_rows:
                n_skipped += 1
                continue

            ranked = torch.argsort(sims_all[gi], descending=True).tolist()
            head = ranked[:top_k]

            gains = np.array([gains_by_row.get(row, 0.0) for row in head])
            ideal = np.zeros(top_k)
            best_gains = sorted(gains_by_row.values(), reverse=True)[:top_k]
            ideal[: len(best_gains)] = best_gains
            dcg = np.cumsum(gains * discount)
            idcg = np.cumsum(ideal * discount)
            ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)

            found = np.cumsum([1.0 if row in relevant_rows else 0.0 for row in head])

            rr = 0.0
            for rank, row in enumerate(ranked, start=1):
                if row in relevant_rows:
                    rr = 1.0 / rank
                    break

            for i, k in enumerate(ks):
                values[f"ndcg@{k}"].append(float(ndcg[i]))
                values[f"hit@{k}"].append(1.0 if found[i] > 0 else 0.0)
            values[f"recall@{top_k}"].append(float(found[-1]) / len(relevant_rows))
            values["mrr"].append(rr)
            # Purely diagnostic: what share of the returned head the teacher ever saw.
            # A drop here means nDCG is grading an increasingly out-of-pool ranking.
            values[f"judged@{top_k}"].append(
                sum(1.0 for row in head if row in gains_by_row) / top_k
            )
            personas.append(group.get("persona_id") or "unknown")

        if n_skipped and warn:
            logger.warning(
                "%d/%d val groups had no chunk_id matching the corpus and were skipped.",
                n_skipped,
                len(groups),
            )

        persona_arr = np.array(personas)
        metrics: dict[str, Any] = {}
        for name in names:
            arr = np.array(values[name])
            metrics[name] = {"overall": float(arr.mean()) if arr.size else 0.0}
            for persona_id in sorted(set(personas)):
                sel = arr[persona_arr == persona_id]
                metrics[name][persona_id] = float(sel.mean()) if sel.size else 0.0

        # Rounded: this lands in training_log.json once per epoch, and 4 dp is well past
        # the resolution of a 276-query mean.
        per_query = {"persona_ids": personas} | {
            name: [round(v, 4) for v in values[name]] for name in names
        }
        return metrics, per_query

    metrics, per_query = score([g["query"] for g in groups], None, warn=True)

    if swap_personas:
        # Rotation, not shuffling: it reassigns every group to a different persona while
        # leaving the persona counts untouched, so the swapped and matched numbers are
        # aggregated over identically sized cells and the difference is attributable to
        # the prefix alone. Groups whose persona is unknown are left as they are.
        swapped = [
            format_query(g["raw_query"], PERSONA_ROTATION.get(g["persona_id"], g["persona_id"]))
            for g in groups
        ]
        swap_metrics, swap_per_query = score(swapped, "eval: persona-swap", warn=False)
        metrics["persona_swap"] = {"metrics": swap_metrics, "per_query": swap_per_query}

    return metrics, per_query


def is_better(candidate: dict, current_best: dict | None, top_k: int) -> bool:
    """Best = highest overall Recall@K; ties → lower val KD loss; ties → higher MRR.

    Selecting on KD loss alone would reward an encoder that matches the teacher's
    distribution while retrieving worse — the reward-hacking guard from the methodology.
    """
    if current_best is None:
        return True
    key = f"recall@{top_k}"
    if candidate[key]["overall"] != current_best[key]["overall"]:
        return candidate[key]["overall"] > current_best[key]["overall"]
    if candidate["val_loss"] != current_best["val_loss"]:
        return candidate["val_loss"] < current_best["val_loss"]
    return candidate["mrr"]["overall"] > current_best["mrr"]["overall"]


def validate(
    model,
    val_loader,
    device: torch.device,
    step_fn,
    temps,
    amp_dtype,
    use_amp,
    micro_batch,
    *,
    tokenizer,
    eval_groups,
    corpus_texts,
    chunk_id_to_row,
    top_k: int,
    relevance_top_m: int,
    eval_batch_size: int,
    max_length: int,
    can_eval_retrieval: bool,
    is_main: bool,
    doc_base_only: bool = False,
    swap_personas: bool = False,
) -> tuple[float, dict | None, dict | None]:
    """Val KD loss on **every** rank; retrieval metrics on rank 0 only.

    Every rank must call this: ``eval_epoch`` all-reduces, so a rank that skips it
    leaves the others blocked on a collective forever — a hang with no traceback.
    Retrieval metrics are rank-0 only and the caller must ``dist.barrier()`` after,
    so other ranks cannot race ahead into the next epoch mid-save.
    """
    val_loss = eval_epoch(
        model, val_loader, device, step_fn, temps, amp_dtype, use_amp, micro_batch
    )
    if not (is_main and can_eval_retrieval):
        return val_loss, None, None

    metrics, per_query = evaluate_retrieval(
        unwrap(model),
        tokenizer,
        eval_groups,
        corpus_texts,
        chunk_id_to_row,
        device,
        top_k,
        relevance_top_m,
        eval_batch_size,
        max_length,
        amp_dtype,
        use_amp,
        doc_base_only,
        swap_personas,
    )
    metrics["val_loss"] = val_loss
    return val_loss, metrics, per_query


def log_swap_block(metrics: dict, top_k: int) -> None:
    """One line contrasting persona-matched retrieval with the rotated-persona control.

    The delta *is* the personalisation signal. Near zero means the encoder is ignoring
    the ``Instruct:`` prefix and every headline gain is generic retrieval quality —
    the outcome the thesis has to rule out, so it is logged every epoch rather than
    reconstructed afterwards.
    """
    swap = metrics.get("persona_swap")
    if not swap:
        return
    sm = swap["metrics"]
    logger.info(
        "  persona-swap  nDCG@1 %.4f (%+.4f) | nDCG@%d %.4f (%+.4f) | MRR %.4f (%+.4f)",
        sm["ndcg@1"]["overall"],
        sm["ndcg@1"]["overall"] - metrics["ndcg@1"]["overall"],
        top_k,
        sm[f"ndcg@{top_k}"]["overall"],
        sm[f"ndcg@{top_k}"]["overall"] - metrics[f"ndcg@{top_k}"]["overall"],
        sm["mrr"]["overall"],
        sm["mrr"]["overall"] - metrics["mrr"]["overall"],
    )


def log_eval_block(header: str, metrics: dict, top_k: int, suffix: str = "") -> None:
    """Curves first, then the headline scalars, then one line per persona."""
    ks = range(1, top_k + 1)
    logger.info("%s%s", header, suffix)
    logger.info(
        "  nDCG@1..%d   %s", top_k, " ".join(f"{metrics[f'ndcg@{k}']['overall']:.3f}" for k in ks)
    )
    logger.info(
        "  Hit@1..%d    %s", top_k, " ".join(f"{metrics[f'hit@{k}']['overall']:.3f}" for k in ks)
    )
    logger.info(
        "  Recall@%d %.4f | MRR %.4f | judged@%d %.4f",
        top_k,
        metrics[f"recall@{top_k}"]["overall"],
        metrics["mrr"]["overall"],
        top_k,
        metrics[f"judged@{top_k}"]["overall"],
    )
    for persona_id in sorted(k for k in metrics["mrr"] if k != "overall"):
        logger.info(
            "    %-9s nDCG@%d %.4f | Hit@%d %.4f | Recall@%d %.4f | MRR %.4f",
            persona_id,
            top_k,
            metrics[f"ndcg@{top_k}"][persona_id],
            top_k,
            metrics[f"hit@{top_k}"][persona_id],
            top_k,
            metrics[f"recall@{top_k}"][persona_id],
            metrics["mrr"][persona_id],
        )


# =============================================================================
# Training entry point
# =============================================================================


def train(config: dict, rank: int = 0, world_size: int = 1, local_rank: int = 0) -> None:
    """Main training routine. Runs single-GPU when *world_size* is 1, DDP otherwise."""
    is_ddp = world_size > 1
    is_main = rank == 0

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_cfg = config["training"]
    emb_cfg = config["embedder"]
    eval_cfg = config["eval"]

    if torch.cuda.is_available() and train_cfg.get("device", "cuda") == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    precision = train_cfg.get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
    # bf16 Tensor Cores need Ampere+ (compute capability >= 8). Kaggle's T4 is Turing,
    # so honour the configured precision instead of hardcoding bf16 as before.
    bf16_ok = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    if precision == "bf16" and not bf16_ok:
        logger.warning("bf16 requested but unsupported on this GPU; falling back to fp16.")
        precision = "fp16"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model_dtype = (
        torch.float32 if device.type == "cpu" else (torch.bfloat16 if bf16_ok else torch.float16)
    )
    # fp16 gradients underflow without loss scaling; bf16 has the range to skip it.
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype is torch.float16) else None

    mode = config.get("mode", "reader_kd")
    fmt = config.get("format", "triplets")
    train_data = Path(config["data"]["train_data"])
    corpus_path = config["data"].get("corpus_path", "data/chunks/corpus.jsonl")
    output_dir = Path(config.get("checkpoint_dir", "./ropg_kd_checkpoints"))

    if mode == "hard_neg":
        suffix = "triplets" if fmt == "triplets" else "pairs"
        train_path = train_data / f"train_{suffix}.jsonl"
        val_path = train_data / f"val_{suffix}.jsonl"
    else:
        train_path = train_data / "train.jsonl"
        val_path = train_data / "val.jsonl"

    if mode == "hard_neg":
        # The trainer never derives triplets — it only validates them. Deriving here would
        # have to guess the label filters, which live in configs/datagen_ropg.yaml, so a
        # missing file would silently produce an *unfiltered* run under a config that says
        # filtered. Derivation belongs to one owner:
        #     python -m data.gen_ropg_data --config configs/datagen_ropg.yaml --derive-only
        # (no API calls, no model load), then ship the files with the dataset.
        #
        # Existence alone is not a sufficient check: TripletDataset slices
        # negatives[:max_negatives], so a file built with fewer negatives than the config
        # asks for trains on fewer, silently and with a normal-looking loss curve.
        # {split}_triplets_meta.json records what each file was built with; its absence
        # means the file predates the sidecar and its provenance is unknown.
        want_negatives = train_cfg.get("max_negatives", 4)
        # Always the triplets sidecar, even in pairs format: derive_triplets writes one
        # sidecar per split covering both outputs, since pairs are a cartesian expansion
        # of the same negatives and share their provenance.
        meta_path = train_data / "train_triplets_meta.json"
        fix = (
            "Re-derive with:\n"
            "    python -m data.gen_ropg_data --config configs/datagen_ropg.yaml --derive-only\n"
            "then upload the resulting *_triplets.jsonl / *_pairs.jsonl / "
            "*_triplets_meta.json alongside train.jsonl."
        )
        if not train_path.exists():
            raise FileNotFoundError(f"{train_path} not found (mode=hard_neg).\n{fix}")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found, so {train_path.name} has unknown provenance "
                f"(it predates the sidecar).\n{fix}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        have = meta.get("max_negatives", -1)
        if have < want_negatives:
            raise ValueError(
                f"{train_path.name} was built with max_negatives={have}, but "
                f"training.max_negatives={want_negatives}. Training would silently use "
                f"{have} negatives per query.\n{fix}"
            )

        # Provenance: every run states which triplet build it trained on, so a result can
        # be traced back to its filter settings without inspecting the data directory.
        if is_main:
            logger.info(
                "Triplets: max_negatives=%s | filters=%s | groups %s/%s retained",
                have,
                meta.get("filters", {}).get("enabled"),
                meta.get("groups_retained"),
                meta.get("groups_total"),
            )

    batch_size = train_cfg["batch_size"]
    num_epochs = train_cfg["epochs"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    micro_batch = train_cfg.get("doc_micro_batch", 2)
    max_negatives = train_cfg.get("max_negatives", 4)
    max_documents = train_cfg.get("max_documents", 20)
    temps = (
        train_cfg.get("student_temp", 0.05),
        train_cfg.get("teacher_temp", 0.2),
    )
    model_name = emb_cfg["model"]
    max_length = emb_cfg.get("max_seq_length", 2048)
    top_k = eval_cfg.get("top_k", 5)
    relevance_top_m = eval_cfg.get("relevance_top_m", 3)
    eval_batch_size = eval_cfg.get("eval_batch_size", 8)
    # Cheap: it reuses the corpus embedding and only re-encodes 276 queries. On by
    # default because a run without it cannot tell a personalisation gain from a
    # generic retrieval gain, and that distinction is the thesis claim.
    swap_personas = bool(eval_cfg.get("persona_swap", True))

    if is_main:
        logger.info(
            "Device: %s | world_size: %d | precision: %s | amp: %s | scaler: %s",
            device,
            world_size,
            precision,
            use_amp,
            scaler is not None,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Qwen3-Embedding is a decoder; left padding keeps the final real token last,
    # which is what last_token_pool reads.
    tokenizer.padding_side = "left"

    if mode == "hard_neg":
        if fmt == "triplets":
            train_ds = TripletDataset(str(train_path), max_negatives)
            val_ds = TripletDataset(str(val_path), max_negatives) if val_path.exists() else None
            collate_fn = lambda b: collate_triplets(b, tokenizer, max_length)  # noqa: E731
        else:
            train_ds = PairDataset(str(train_path))
            val_ds = PairDataset(str(val_path)) if val_path.exists() else None
            collate_fn = lambda b: collate_pairs(b, tokenizer, max_length)  # noqa: E731
        # step_fn is built after the model exists: the anchor cache needs a forward
        # pass with the adapter disabled, so it cannot be constructed here.
        step_fn = None
    else:
        train_ds = ScoredDataset(str(train_path), max_documents)
        val_ds = ScoredDataset(str(val_path), max_documents) if val_path.exists() else None
        collate_fn = lambda b: collate_scored(b, tokenizer, max_length)  # noqa: E731
        step_fn = _kd_step

    # Corpus-level retrieval metrics need chunk_ids, which only the scored format
    # carries. In hard_neg mode the scored val file supplies them.
    scored_val_path = train_data / "val.jsonl"
    eval_groups = (
        ScoredDataset(str(scored_val_path), max_documents).items
        if scored_val_path.exists()
        else []
    )
    corpus_texts, chunk_id_to_row = (
        load_corpus(corpus_path) if Path(corpus_path).exists() else ([], {})
    )
    can_eval_retrieval = bool(eval_groups and corpus_texts)
    if is_main and not can_eval_retrieval:
        logger.warning(
            "Corpus-level retrieval eval disabled (corpus=%s, val groups=%d); "
            "checkpoint selection falls back to val loss.",
            corpus_path,
            len(eval_groups),
        )

    if is_main:
        logger.info("Train samples: %d", len(train_ds))
        logger.info("Val samples  : %d", len(val_ds) if val_ds else 0)
        logger.info("Corpus chunks: %d", len(corpus_texts))

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if (is_ddp and val_ds)
        else None
    )
    num_workers = train_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        if val_ds
        else None
    )

    model = QwenEmbeddingModel(
        model_name,
        use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        model_dtype=model_dtype,
        lora_config=config.get("lora"),
    ).to(device)
    if is_main and config.get("lora"):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total
        )

    # Anchoring: build the base-embedding cache and the hard_neg step function now that
    # the model exists. Done before the DDP wrap because it is a pure per-rank forward —
    # no collectives — and every rank needs the identical cache anyway.
    anchor_cfg = config.get("anchor") or {}
    # Anchoring is a hard_neg construct: reader_kd's step function does not consume the
    # cache, so leaving the mode set there would only mislead the eval path into
    # base-encoding the corpus for a run that never froze the document tower.
    anchor_mode = anchor_cfg.get("mode", "none") if mode == "hard_neg" else "none"
    if step_fn is None:
        cache = None
        if anchor_mode != "none" and fmt == "triplets":
            cache = build_anchor_cache(
                model,
                tokenizer,
                [ds for ds in (train_ds, val_ds) if ds is not None],
                device,
                eval_batch_size,
                max_length,
                amp_dtype,
                use_amp,
            )
        elif anchor_mode != "none":
            # PairDataset has no negatives list, so the cache annotation would not apply.
            logger.warning("anchor.mode=%s ignored: format=pairs is unanchored", anchor_mode)
            anchor_mode = "none"
        if is_main:
            logger.info(
                "Anchor: mode=%s lambda_doc=%s lambda_query=%s",
                anchor_mode,
                anchor_cfg.get("lambda_doc", 0.0),
                anchor_cfg.get("lambda_query", 0.0),
            )
        step_fn = make_mnrl_step({**anchor_cfg, "mode": anchor_mode}, cache)

    if is_ddp:
        # static_graph=True lets the same parameters be used by several forwards per
        # step (doc micro-batching) without DDP raising "marked ready twice", and is
        # also what makes non-reentrant gradient checkpointing safe under DDP.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * train_cfg.get("warmup_ratio", 0.1)), total_steps
    )
    if is_main:
        logger.info("Steps/epoch: %d | total: %d", steps_per_epoch, total_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "checkpoint-best"
    best: dict | None = None
    best_epoch = 0
    epoch_metrics: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None

    # Epoch 0: the untrained model. PEFT zero-initializes lora_B and eval() disables
    # dropout, so the adapter is exactly the identity here - these numbers *are* the
    # frozen Qwen3-Embedding baseline (Rung 1), measured through the same val set,
    # relevance definition and code path as every trained epoch. It also exercises the
    # whole eval path (corpus encoding, all-reduce, barrier) in minutes rather than
    # after a two-hour training epoch. Deliberately not eligible for checkpoint-best:
    # an identity adapter winning would silently make Rung 3 equal Rung 1.
    if val_loader:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        base_loss, base_metrics, base_per_query = validate(
            model,
            val_loader,
            device,
            step_fn,
            temps,
            amp_dtype,
            use_amp,
            micro_batch,
            tokenizer=tokenizer,
            eval_groups=eval_groups,
            corpus_texts=corpus_texts,
            chunk_id_to_row=chunk_id_to_row,
            top_k=top_k,
            relevance_top_m=relevance_top_m,
            eval_batch_size=eval_batch_size,
            max_length=max_length,
            can_eval_retrieval=can_eval_retrieval,
            is_main=is_main,
            doc_base_only=anchor_mode == "doc_frozen",
            swap_personas=swap_personas,
        )
        if is_main:
            # No train_loss key: its absence is how the results cell recognises epoch 0.
            baseline = {"epoch": 0, "val_loss": base_loss}
            if base_metrics is not None:
                baseline.update({k: v for k, v in base_metrics.items() if k != "val_loss"})
                baseline["per_query"] = base_per_query
                log_eval_block(
                    f"Baseline (untrained, epoch 0) | val {base_loss:.4f}", base_metrics, top_k
                )
                log_swap_block(base_metrics, top_k)
            else:
                logger.info("Baseline (untrained, epoch 0) | val %.4f", base_loss)
            if device.type == "cuda":
                logger.info(
                    "Epoch 0 peak GPU: allocated %.2f GiB (eval only)",
                    torch.cuda.max_memory_allocated(device) / 2**30,
                )
            epoch_metrics.append(baseline)
        if is_ddp:
            dist.barrier()

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            step_fn,
            temps,
            amp_dtype,
            use_amp,
            grad_clip,
            micro_batch,
            is_main,
        )
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        if val_loader:
            val_loss, metrics, per_query = validate(
                model,
                val_loader,
                device,
                step_fn,
                temps,
                amp_dtype,
                use_amp,
                micro_batch,
                tokenizer=tokenizer,
                eval_groups=eval_groups,
                corpus_texts=corpus_texts,
                chunk_id_to_row=chunk_id_to_row,
                top_k=top_k,
                relevance_top_m=relevance_top_m,
                eval_batch_size=eval_batch_size,
                max_length=max_length,
                can_eval_retrieval=can_eval_retrieval,
                is_main=is_main,
                doc_base_only=anchor_mode == "doc_frozen",
                swap_personas=swap_personas,
            )
            entry["val_loss"] = val_loss

            # Checkpointing happens on rank 0 only; the barrier below stops other ranks
            # racing into the next epoch mid-save.
            if is_main:
                if metrics is not None:
                    entry.update({k: v for k, v in metrics.items() if k != "val_loss"})
                    entry["per_query"] = per_query
                    if is_better(metrics, best, top_k):
                        best, best_epoch = metrics, epoch
                        unwrap(model).save_adapter(save_dir)
                        tokenizer.save_pretrained(save_dir)
                    log_eval_block(
                        f"Epoch {epoch}/{num_epochs} | train {train_loss:.4f} | "
                        f"val {val_loss:.4f}",
                        metrics,
                        top_k,
                        "  *best*" if best_epoch == epoch else "",
                    )
                    log_swap_block(metrics, top_k)
                else:
                    if best is None or val_loss < best["val_loss"]:
                        best, best_epoch = {"val_loss": val_loss}, epoch
                        unwrap(model).save_adapter(save_dir)
                        tokenizer.save_pretrained(save_dir)
                    logger.info(
                        "Epoch %d/%d | train %.4f | val %.4f%s",
                        epoch,
                        num_epochs,
                        train_loss,
                        val_loss,
                        "  *best (loss)*" if best_epoch == epoch else "",
                    )
            if is_ddp:
                dist.barrier()
        elif is_main:
            logger.info("Epoch %d/%d | train %.4f", epoch, num_epochs, train_loss)

        if device.type == "cuda":
            # Logged on *every* rank, not just rank 0: peak memory is data-dependent
            # (the attention buffer scales with rows x width^2, and the sampler reshuffles
            # each epoch), so the rank that OOMs is not necessarily the one that reports.
            # A flat series across epochs means the peak is a property of the worst batch;
            # a climbing one means something is actually being retained.
            entry["peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            logger.info(
                "Epoch %d peak GPU: allocated %.2f GiB | reserved %.2f GiB",
                epoch,
                entry["peak_gib"],
                torch.cuda.max_memory_reserved(device) / 2**30,
            )

        epoch_metrics.append(entry)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if is_main and baseline is not None and best is not None:
        # The question the baseline exists to answer, stated once at the end. A run that
        # never beats the untrained encoder is a real result and belongs in the log at
        # WARNING, not buried in a table of per-epoch numbers.
        improved = False
        # Headline pair first — see docs/experiment-design.md, "Stage-1 retriever runs".
        # nDCG@1 grades the single slot where the teacher's SNR is high; Recall@K is what
        # is_better already selects on. nDCG@K trails because over half its mass sits in
        # slots graded by teacher gaps below the label noise floor.
        for name in ("ndcg@1", f"recall@{top_k}", "mrr", f"ndcg@{top_k}"):
            if name not in baseline or name not in best:
                continue
            before, after = baseline[name]["overall"], best[name]["overall"]
            logger.info(
                "%-9s baseline %.4f -> best (epoch %d) %.4f   %+.4f",
                name,
                before,
                best_epoch,
                after,
                after - before,
            )
            improved |= after > before
        if not improved:
            logger.warning(
                "No epoch beat the untrained baseline on any headline metric. "
                "checkpoint-best is still a trained adapter, not the baseline - treat "
                "this as a negative result, not a checkpoint to ship."
            )

    if is_main:
        final_dir = output_dir / "checkpoint-final"
        unwrap(model).save_adapter(final_dir)
        tokenizer.save_pretrained(final_dir)
        (output_dir / "training_log.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "seed": seed,
                    "world_size": world_size,
                    "epoch_metrics": epoch_metrics,
                    "best_epoch": best_epoch,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Best epoch %d → %s | final → %s", best_epoch, save_dir, final_dir)


# =============================================================================
# Launchers
# =============================================================================


def main_worker(config: dict) -> None:
    """Single training process. Launched once per GPU by ``torchrun``.

    Rank information comes from the environment, which is torchrun's contract and
    also degrades correctly to one process under a bare ``python -m rl.ropg_kd``.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [rank{rank}] %(message)s",
        force=True,
    )
    if world_size > 1:
        # torchrun sets these; the defaults only matter under a manual launch.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            # NCCL's default is ~30 minutes, long enough that a rendezvous stall
            # reads as an infinite hang. Fail loudly instead: a healthy two-T4
            # handshake takes seconds.
            timeout=timedelta(minutes=10),
        )
    try:
        train(config, rank=rank, world_size=world_size, local_rank=local_rank)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="ROPG-KD: fine-tune the Qwen3-Embedding encoder with KD."
    )
    parser.add_argument("--config", required=True, help="Path to train_ropg.yaml.")
    parser.add_argument("--train_data", default=None, help="Override data.train_data.")
    parser.add_argument("--corpus_path", default=None, help="Override data.corpus_path.")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint_dir.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.train_data:
        config["data"]["train_data"] = args.train_data
    if args.corpus_path:
        config["data"]["corpus_path"] = args.corpus_path
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir

    # One rank per process. Under torchrun this file is executed once per GPU and
    # main_worker picks its rank up from the environment; run bare, it is a single
    # process with world_size 1.
    main_worker(config)


if __name__ == "__main__":
    main()
