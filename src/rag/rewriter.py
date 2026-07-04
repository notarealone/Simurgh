"""Persona-conditioned query rewriter (prompt-based, untrained)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.llm import OpenAICompatClient

_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a Persian educational RAG system. "
    "Given a learner profile and an original question, rewrite the question as a "
    "retrieval query that will surface the most pedagogically useful passages for "
    "that specific learner. "
    "Rules: output ONLY the rewritten query — no explanation, no preamble, no quotes. "
    "Keep it in Persian if the original is Persian. "
    "You may expand abbreviations, add prerequisite terms, or rephrase for clarity, "
    "but do not invent facts or change the question's intent."
)


def _build_rewrite_prompt(profile_rendered: str, query: str) -> list[dict[str, str]]:
    user = (
        f"Learner profile: {profile_rendered}\n\n"
        f"Original question: {query}\n\n"
        "Rewritten retrieval query:"
    )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


class PromptedRewriter:
    """Calls the LLM to produce a persona-conditioned rewrite of the input query."""

    def __init__(self, llm: OpenAICompatClient) -> None:
        self.llm = llm

    def rewrite(self, profile_rendered: str, query: str) -> str:
        """Return a persona-conditioned rewrite of *query*."""
        messages = _build_rewrite_prompt(profile_rendered, query)
        return self.llm.chat(messages).strip()
