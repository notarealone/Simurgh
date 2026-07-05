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
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
        batch_size: int = 32,
        fp16: bool = False,
    ) -> None:
        model_kwargs = {"torch_dtype": torch.float16} if fp16 else {}
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True, model_kwargs=model_kwargs
        )
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
