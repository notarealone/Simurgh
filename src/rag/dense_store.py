"""FAISS-backed dense vector store for the RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import faiss

from data.persian import normalize

if TYPE_CHECKING:
    from data.chunking import Chunk
    from rag.embedder import Qwen3Embedder
    from rag.store import Hit


class DenseStore:
    """Persistent dense retrieval store: BGE-M3 encoder + FAISS IndexFlatIP."""

    def __init__(
        self,
        index_path: str | Path,
        meta_path: str | Path,
        embedder: Qwen3Embedder,
    ) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedder = embedder
        self._meta: list[dict] = []

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self._index = faiss.IndexFlatIP(self.embedder.dim)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        texts = [normalize(chunk.text) for chunk in chunks]
        vecs = self.embedder.encode(texts)
        self._index.add(vecs)
        for chunk in chunks:
            self._meta.append(
                {"raw_text": chunk.text, "source": chunk.source, "chunk_index": chunk.index}
            )
        self._save()

    def search(self, query: str, top_k: int = 5, instruction: str = "") -> list[Hit]:
        from rag.store import Hit

        if self._index.ntotal == 0:
            return []
        vec = self.embedder.encode_query([normalize(query)], instruction=instruction)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            m = self._meta[idx]
            results.append(
                Hit(
                    raw_text=m["raw_text"],
                    source=m["source"],
                    chunk_index=m["chunk_index"],
                    score=float(score),
                )
            )
        return results

    def count(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self.embedder.dim)
        self._meta = []
        self._save()

    def close(self) -> None:
        self._save()

    def _save(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")

    def __enter__(self) -> DenseStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
