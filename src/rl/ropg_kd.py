"""ROPG-KD: fine-tune the BGE-M3 encoder via knowledge distillation from LLM judge scores.

Stage 1 of Simurgh's two-stage training:
  1. Load scored (query, persona, document) triples from the scorer output.
  2. Group by (query, persona_id) to form soft target distributions over top-K docs.
  3. Fine-tune the BGE-M3 encoder with a LoRA adapter to minimise KL divergence
     between its similarity distribution and the teacher's utility distribution.
  4. Checkpoint on validation Recall@K per persona.

Runs on GPU (Kaggle / university cluster). Install the training extra first:
    uv sync --extra embedding --extra training

Typical usage:
    uv run python -m rl.ropg_kd --config configs/phase3_ropg_kd.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_scored_triples(path: str | Path) -> dict[tuple[str, str], list[dict]]:
    """Load scored triples and group by (query, persona_id).

    Returns:
        dict mapping (query_text, persona_id) -> list of {doc_text, score} sorted by doc_idx.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("score") is None:
            continue
        key = (rec["query"], rec["persona_id"])
        groups.setdefault(key, []).append(
            {
                "doc_text": rec["doc_text"],
                "score": float(rec["score"]),
                "doc_idx": rec["doc_idx"],
            }
        )
    # Sort each group by doc_idx to ensure consistent ordering.
    for key in groups:
        groups[key].sort(key=lambda x: x["doc_idx"])
    return groups


# ── Training ──────────────────────────────────────────────────────────────────


def train(config: dict) -> None:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    from personalization.profiles import render_profile

    kd_cfg = config["ropg_kd"]
    scored_path = config["scorer"]["output_path"]
    checkpoint_dir = Path(kd_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device = kd_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load scored groups.
    groups = load_scored_triples(scored_path)
    items = list(groups.items())
    logger.info("Loaded %d (query, persona) groups from %s", len(items), scored_path)

    # Load the encoder and wrap with a LoRA adapter.
    emb_cfg = config.get("embedder", {})
    model_name = emb_cfg.get("model", "BAAI/bge-m3")
    st_model = SentenceTransformer(model_name, device=device)

    # Access the underlying transformer for LoRA.
    transformer = st_model[0].auto_model
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=kd_cfg.get("lora_r", 8),
        lora_alpha=kd_cfg.get("lora_alpha", 16),
        lora_dropout=kd_cfg.get("lora_dropout", 0.1),
        target_modules=kd_cfg.get("lora_target_modules", ["query", "value"]),
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=kd_cfg.get("lr", 2e-4),
        weight_decay=kd_cfg.get("weight_decay", 0.01),
    )

    kd_temp = kd_cfg.get("kd_temperature", 1.0)
    n_epochs = kd_cfg.get("epochs", 3)
    batch_size = kd_cfg.get("batch_size", 8)

    def _last_token_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Last-token pooling for decoder-based embedding models (Qwen3-Embedding)."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    def encode_docs(texts: list[str]) -> torch.Tensor:
        """Encode document texts (no instruction prefix), L2-normalised."""
        tokenizer = st_model.tokenizer
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = transformer(**enc)
        pooled = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(pooled, p=2, dim=-1)

    def encode_query(text: str, instruction: str) -> torch.Tensor:
        """Encode a single query with persona instruction prefix, L2-normalised."""
        formatted = f"Instruct: {instruction}\nQuery: {text}"
        tokenizer = st_model.tokenizer
        enc = tokenizer(
            [formatted], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = transformer(**enc)
        pooled = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(pooled, p=2, dim=-1)

    best_loss = math.inf
    for epoch in range(n_epochs):
        random.shuffle(items)
        epoch_loss = 0.0
        n_batches = 0
        transformer.train()

        for i in tqdm(range(0, len(items), batch_size), desc=f"Epoch {epoch + 1}/{n_epochs}"):
            batch = items[i : i + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for (query, persona_id), docs in batch:
                if len(docs) < 2:
                    continue
                profile_rendered = render_profile(persona_id)
                doc_texts = [d["doc_text"] for d in docs]
                scores = torch.tensor([d["score"] for d in docs], device=device)

                q_vec = encode_query(query, instruction=profile_rendered)  # (1, dim)
                d_vecs = encode_docs(doc_texts)  # (K, dim)
                sims = (q_vec @ d_vecs.T).squeeze(0)  # (K,)

                soft_targets = F.softmax(scores / kd_temp, dim=0)
                log_probs = F.log_softmax(sims / kd_temp, dim=0)
                # KL(teacher || student) = sum(teacher * (log teacher - log student))
                kl = F.kl_div(log_probs, soft_targets, reduction="sum")
                batch_loss = batch_loss + kl

            if batch_loss.item() > 0:
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d | avg KL loss: %.4f", epoch + 1, n_epochs, avg_loss)

        ckpt_path = checkpoint_dir / f"ropg_kd_epoch{epoch + 1}"
        transformer.save_pretrained(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "ropg_kd_best"
            transformer.save_pretrained(str(best_path))
            logger.info("Best checkpoint updated: %s", best_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="ROPG-KD: fine-tune Qwen3-Embedding encoder with KD."
    )
    parser.add_argument("--config", required=True, help="Path to phase3 YAML config.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
