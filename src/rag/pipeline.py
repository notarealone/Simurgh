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
    def __init__(self, config: dict) -> None:
        self.config = config
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

    def retrieve(self, query: str) -> list[Hit]:
        """Retrieve top-k hits WITHOUT calling the LLM (no API key needed)."""
        return self.kb.search(query, self.top_k)

    def ask(self, query: str, variant: str | None = None) -> RAGResult:
        """Retrieve context, build the prompt, and call the LLM.

        *variant* overrides the configured prompt language ("en" or "fa") for this call.
        """
        hits = self.retrieve(query)
        messages = build_rag_prompt(query, hits, variant=variant or self.prompt_variant)
        answer = self.llm.chat(messages)
        return RAGResult(answer=answer, hits=hits, messages=messages)

    def close(self) -> None:
        self.kb.close()
