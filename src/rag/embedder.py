"""Qwen3-Embedding-0.6B dense encoder wrapper."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Qwen3Embedder:
    """Encodes text to L2-normalised float32 dense vectors using Qwen3-Embedding.

    Qwen3-Embedding is a decoder-based model that uses last-token pooling and
    supports instruction-prefixed queries for task-conditioned retrieval.
    sentence-transformers handles last-token pooling automatically via the model's
    config when trust_remote_code=True.

    If *adapter_path* is set, a LoRA adapter (e.g. the ROPG-KD checkpoint from
    src/rl/ropg_kd.py) is loaded on top of the base model and merged into its weights.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
        batch_size: int = 32,
        fp16: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        model_kwargs = {"torch_dtype": torch.float16} if fp16 else {}
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True, model_kwargs=model_kwargs
        )
        if adapter_path is not None:
            from peft import PeftModel

            # from_pretrained injects the adapter into the base model in place and
            # merge_and_unload folds the LoRA weights into it, so no reassignment is
            # needed (auto_model is a read-only property on the Transformer module).
            PeftModel.from_pretrained(self.model[0].auto_model, adapter_path).merge_and_unload()
        self.batch_size = batch_size
        self.dim: int = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode documents (no instruction prefix). Returns float32 (N, dim), L2-normalised."""
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, texts: list[str], instruction: str = "") -> np.ndarray:
        """Encode queries with an optional persona instruction prefix.

        When *instruction* is provided the model sees:
            ``Instruct: {instruction}\\nQuery: {text}``
        which lets it condition the embedding on the learner profile.
        """
        if instruction:
            prompt = f"Instruct: {instruction}\nQuery: "
            vecs = self.model.encode(
                texts,
                prompt=prompt,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vecs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.array(vecs, dtype=np.float32)
