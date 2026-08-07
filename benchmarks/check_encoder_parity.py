"""Verify the ROPG-KD training encoder matches the inference encoder exactly.

This is the gate on ROPG-KD being worth anything. ``rl.ropg_kd.QwenEmbeddingModel``
(training) and ``rag.embedder.Qwen3Embedder`` (serving, via sentence-transformers)
must produce the same vector for the same text. If they diverge, a LoRA adapter
trained through one is read out through the other and most of the fine-tune is lost —
silently, with no error and a perfectly healthy-looking loss curve.

Three things have to agree: pooling (last-token), the query template
(``Instruct: {persona}\\nQuery: {text}``), and tokenization (special tokens +
max_seq_length). This script checks all three at once and, on failure, reports which.

Run on a GPU box after `uv sync --extra embedding --extra training`:
    uv run python benchmarks/check_encoder_parity.py --config configs/train_ropg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from personalization.profiles import PERSONAS
from rag.embedder import Qwen3Embedder
from rl.ropg_kd import QwenEmbeddingModel, format_query

# Below this, the two paths are computing genuinely different things.
THRESHOLD = 0.999


def load_samples(corpus_path: str, n: int) -> list[str]:
    texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


@torch.no_grad()
def encode_training_path(
    model, tokenizer, texts: list[str], device: str, max_length: int
) -> np.ndarray:
    tokens = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    return model(tokens["input_ids"], tokens["attention_mask"]).float().cpu().numpy()


def report(label: str, a: np.ndarray, b: np.ndarray) -> float:
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    worst = float(cos.min())
    status = "PASS" if worst >= THRESHOLD else "FAIL"
    print(f"  [{status}] {label:<28} min={worst:.6f}  mean={float(cos.mean()):.6f}")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ropg.yaml")
    ap.add_argument("--n", type=int, default=20, help="Samples per comparison.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = cfg["embedder"]["model"]
    max_length = cfg["embedder"].get("max_seq_length", 2048)
    corpus_path = cfg["data"].get("corpus_path", "data/chunks/corpus.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"model={model_name}  device={device}  max_seq_length={max_length}\n")

    docs = load_samples(corpus_path, args.n)
    persona_id = next(iter(PERSONAS))
    raw_queries = [d.split("\n")[0][:200] for d in docs]
    queries = [format_query(q, persona_id) for q in raw_queries]

    # ---- inference path -----------------------------------------------------
    from personalization.profiles import render_profile

    infer = Qwen3Embedder(model_name, device=device, batch_size=8, max_seq_length=max_length)
    infer_docs = infer.encode(docs)
    infer_queries = infer.encode_query(raw_queries, instruction=render_profile(persona_id))
    del infer
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- training path ------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    train_model = QwenEmbeddingModel(
        model_name,
        use_gradient_checkpointing=False,
        model_dtype=torch.float32 if device == "cpu" else torch.float16,
        lora_config=None,
    ).to(device)
    train_model.eval()

    train_docs = encode_training_path(train_model, tokenizer, docs, device, max_length)
    train_queries = encode_training_path(train_model, tokenizer, queries, device, max_length)

    print("Parity (training encoder vs. Qwen3Embedder):")
    worst_docs = report("documents (bare)", infer_docs, train_docs)
    worst_q = report("queries (Instruct template)", infer_queries, train_queries)

    if min(worst_docs, worst_q) >= THRESHOLD:
        print("\nPASS - encoders agree. ROPG-KD adapters will transfer to serving.")
        return 0

    # ---- diagnosis ----------------------------------------------------------
    print("\nFAIL - the two paths disagree. Narrowing it down:\n")

    if worst_docs < THRESHOLD <= worst_q:
        print("  Documents differ but queries match -> not a template problem.")
    if worst_q < THRESHOLD <= worst_docs:
        print("  Queries differ but documents match -> the Instruct template or its")
        print("  prompt-masking differs. Check format_query() against")
        print("  Qwen3Embedder.encode_query.")

    # The most common cause: sentence-transformers appends EOS, AutoTokenizer may not.
    ids = tokenizer(docs[:1], truncation=True, max_length=max_length)["input_ids"][0]
    eos_id = tokenizer.eos_token_id
    print(f"\n  AutoTokenizer last token id: {ids[-1]}  (eos_token_id={eos_id})")
    if eos_id is not None and ids[-1] != eos_id:
        print("  -> AutoTokenizer is NOT appending EOS. Since pooling reads the final")
        print("     token, this alone changes every vector. Fix in rl.ropg_kd by")
        print("     appending tokenizer.eos_token to each text before tokenizing,")
        print("     then re-run this check.")
    else:
        print("  -> EOS handling looks consistent; inspect pooling and max_seq_length next.")

    st_max = getattr(Qwen3Embedder, "max_seq_length", None)
    print(f"  Configured max_seq_length={max_length} (st reports {st_max}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
