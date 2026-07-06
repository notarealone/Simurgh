from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from data.chunking import chunk_text
from data.loaders import load_document
from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL
from rag.llm import OpenAICompatClient
from rag.prompts import build_rag_prompt
from rag.store import KnowledgeBase

if TYPE_CHECKING:
    from rag.store import Hit


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config and seed the RNG (.env is loaded by data.settings)."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("seed") is not None:
        random.seed(config["seed"])
    return config


@dataclass
class RAGResult:
    answer: str
    hits: list[Hit]
    messages: list[dict[str, str]]


class NaiveRAG:
    def __init__(
        self,
        config: dict,
        store: object | None = None,
        rewriter: object | None = None,
    ) -> None:
        self.config = config
        if store is not None:
            self.kb = store
        else:
            self.kb = KnowledgeBase(config["knowledge_base"]["db_path"])
        self.top_k = config["retrieval"]["top_k"]
        self.chunk_size = config["chunking"]["chunk_size"]
        self.chunk_overlap = config["chunking"]["chunk_overlap"]
        llm = config["llm"]
        self.llm = OpenAICompatClient(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=llm["model"],
            temperature=llm["temperature"],
            max_tokens=llm["max_tokens"],
        )
        self.prompt_variant = llm.get("prompt_variant", "en")
        self.rewriter = rewriter

    def ingest(self, paths: list[str | Path]) -> int:
        """Load, chunk, and index every document in *paths*; return total chunk count."""
        total = 0
        for path in paths:
            text = load_document(path)
            chunks = chunk_text(
                text,
                source=Path(path).name,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            self.kb.add_chunks(chunks)
            total += len(chunks)
        return total

    def retrieve(self, query: str, profile_str: str | None = None) -> list[Hit]:
        """Retrieve top-k hits, optionally rewriting *query* for *profile_str* first."""
        effective_query = query
        if self.rewriter is not None and profile_str is not None:
            effective_query = self.rewriter.rewrite(profile_str, query)
        return self.kb.search(effective_query, self.top_k)

    def ask(
        self,
        query: str,
        profile_str: str | None = None,
        variant: str | None = None,
    ) -> RAGResult:
        """Retrieve context, build the prompt, and call the LLM.

        *profile_str* is the rendered persona description; if provided and a rewriter is
        configured, the query is rewritten before retrieval.
        *variant* overrides the configured prompt language ("en" or "fa") for this call.
        """
        hits = self.retrieve(query, profile_str=profile_str)
        messages = build_rag_prompt(query, hits, variant=variant or self.prompt_variant)
        answer = self.llm.chat(messages)
        return RAGResult(answer=answer, hits=hits, messages=messages)

    def close(self) -> None:
        self.kb.close()


class DenseRAG(NaiveRAG):
    """Rung 1: BM25 replaced with Qwen3-Embedding dense retrieval backed by FAISS."""

    def __init__(self, config: dict) -> None:
        from rag.dense_store import DenseStore
        from rag.embedder import Qwen3Embedder

        emb_cfg = config.get("embedder", {})
        embedder = Qwen3Embedder(
            model_name=emb_cfg.get("model", "Qwen/Qwen3-Embedding-0.6B"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=emb_cfg.get("batch_size", 32),
            fp16=emb_cfg.get("fp16", False),
            adapter_path=emb_cfg.get("adapter_path"),
        )
        store = DenseStore(
            index_path=config["knowledge_base"]["index_path"],
            meta_path=config["knowledge_base"]["meta_path"],
            embedder=embedder,
        )
        super().__init__(config, store=store)
        self._embedder = embedder

    def close(self) -> None:
        self.kb.close()


class PersonaRAG(DenseRAG):
    """Rung 2: DenseRAG + persona-conditioned untrained query rewriter."""

    def __init__(self, config: dict) -> None:
        from rag.rewriter import DPORewriter, PromptedRewriter

        rewriter_cfg = config.get("rewriter", {})
        rewriter: PromptedRewriter | DPORewriter | None = None
        if rewriter_cfg.get("enabled", False):
            if rewriter_cfg.get("type") == "dpo":
                rewriter = DPORewriter(
                    model_name=rewriter_cfg.get("model", "Qwen/Qwen3-4B"),
                    adapter_path=rewriter_cfg.get("adapter_path"),
                    device=rewriter_cfg.get("device", "cuda"),
                    max_new_tokens=rewriter_cfg.get("max_new_tokens", 200),
                )
            else:
                rw_llm_cfg = rewriter_cfg.get("llm", config["llm"])
                rw_llm = OpenAICompatClient(
                    base_url=OPENAI_BASE_URL,
                    api_key=OPENAI_API_KEY,
                    model=rw_llm_cfg["model"],
                    temperature=rw_llm_cfg.get("temperature", 0.3),
                    max_tokens=rw_llm_cfg.get("max_tokens", 200),
                )
                rewriter = PromptedRewriter(rw_llm)

        from rag.dense_store import DenseStore
        from rag.embedder import Qwen3Embedder

        emb_cfg = config.get("embedder", {})
        embedder = Qwen3Embedder(
            model_name=emb_cfg.get("model", "Qwen/Qwen3-Embedding-0.6B"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=emb_cfg.get("batch_size", 32),
            fp16=emb_cfg.get("fp16", False),
            adapter_path=emb_cfg.get("adapter_path"),
        )
        store = DenseStore(
            index_path=config["knowledge_base"]["index_path"],
            meta_path=config["knowledge_base"]["meta_path"],
            embedder=embedder,
        )
        NaiveRAG.__init__(self, config, store=store, rewriter=rewriter)
        self._embedder = embedder

    def retrieve(self, query: str, profile_str: str | None = None) -> list[Hit]:
        """Retrieve with query rewriting + persona-conditioned embedding instruction."""
        effective_query = query
        if self.rewriter is not None and profile_str is not None:
            effective_query = self.rewriter.rewrite(profile_str, query)
        instruction = profile_str or ""
        return self.kb.search(effective_query, self.top_k, instruction=instruction)
