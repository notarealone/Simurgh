"""Learner persona definitions for the Simurgh RAG pipeline.

Schema and rendered texts sourced from docs/personas.md.
Three train-split personas (crammer, scholar, steady) are used for both training and
validation; newcomer is test-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    id: str
    comprehension: str  # L1–L4
    prior_knowledge: str  # L1–L4
    learning_goal: str  # L1–L4
    explanation_style: str  # L1–L4
    split: str  # "train" or "test"
    rendered: str  # natural-language prose injected into prompts


PERSONAS: dict[str, Profile] = {
    "crammer": Profile(
        id="crammer",
        comprehension="L1",
        prior_knowledge="L1",
        learning_goal="L1",
        explanation_style="L4",
        split="train",
        rendered=(
            "A ninth-grader who finds the textbook hard to follow and has little background "
            "on this topic. Mainly wants to pass the exam — give the answer and what is needed "
            "to score — but it must be spelled out simply, step by step, with examples."
        ),
    ),
    "scholar": Profile(
        id="scholar",
        comprehension="L4",
        prior_knowledge="L3",
        learning_goal="L4",
        explanation_style="L1",
        split="train",
        rendered=(
            "A ninth-grader who reads dense material easily and has solid background on this "
            "topic. Wants to understand the underlying why and how, and the connections between "
            "ideas. Prefers a terse, high-level treatment without hand-holding or padding."
        ),
    ),
    "steady": Profile(
        id="steady",
        comprehension="L3",
        prior_knowledge="L2",
        learning_goal="L2",
        explanation_style="L2",
        split="train",
        rendered=(
            "A capable ninth-grader with average background on this topic. Wants a correct "
            "answer with a brief justification, balanced toward exam needs. Does not need "
            "elaborate scaffolding, but does appreciate a one-line reason."
        ),
    ),
    "newcomer": Profile(
        id="newcomer",
        comprehension="L3",
        prior_knowledge="L1",
        learning_goal="L4",
        explanation_style="L4",
        split="test",
        rendered=(
            "A bright ninth-grader who reads well but is new to this topic. Wants real "
            "understanding — the why and how, not just the answer — and needs worked examples "
            "to bridge the missing background."
        ),
    ),
}


def render_profile(persona_id: str) -> str:
    """Return the natural-language rendering for *persona_id*."""
    if persona_id not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona_id!r}. Valid: {list(PERSONAS)}")
    return PERSONAS[persona_id].rendered


def train_personas() -> list[Profile]:
    """Return the three training personas (excludes the test holdout)."""
    return [p for p in PERSONAS.values() if p.split == "train"]
