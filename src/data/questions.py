"""Render a question's complete retrieval context — one definition, shared by every stage.

Both data-generation stages and the models they feed must agree on the exact string that
represents a question. The retriever LoRA was distilled on the form produced here
(``Passage:`` / ``Question:`` / ``Options:`` / ``Pairs:`` / ``Items:`` sections), so a
caller that reconstructs a query from ``stem`` alone silently queries the encoder with a
form it never saw in training.

Gold fields (``answer``, ``explanation``) travel beside the query in ``QuestionContext``
and are deliberately *not* part of it: they are judge-only reference material, and
folding them into a retrieval query would leak the answer into the retriever's input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class QuestionContext:
    """A question's retrieval query plus its judge-only gold fields."""

    query: str
    answer: object | None = None
    explanation: str | None = None


def render_question_value(value: object) -> str:
    """Render an extracted value without changing string content."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_numbered_section(label: str, values: object) -> str | None:
    if values is None:
        return None
    entries = values if isinstance(values, list) else [values]
    if not entries:
        return None
    rendered = "\n".join(
        f"{index}. {render_question_value(value)}" for index, value in enumerate(entries, start=1)
    )
    return f"{label}:\n{rendered}"


def load_question(exam_stem: str, qid: str, questions_dir: Path) -> QuestionContext:
    """Load and render the complete retrieval context for *qid*."""
    qfile = questions_dir / f"{exam_stem}.json"
    if not qfile.exists():
        raise FileNotFoundError(f"Question file not found: {qfile}")
    data = json.loads(qfile.read_text(encoding="utf-8"))
    for q in data.get("questions", []):
        if q["id"] == qid:
            sections: list[str] = []
            group_id = q.get("group_id")
            if group_id is not None:
                passages = data.get("passages", data.get("passage", []))
                matching_passage: object | None = None

                if isinstance(passages, dict):
                    if passages.get("id") == group_id or passages.get("group_id") == group_id:
                        matching_passage = passages
                    else:
                        matching_passage = passages.get(group_id)
                        if matching_passage is None:
                            matching_passage = passages.get(str(group_id))
                elif isinstance(passages, list):
                    for passage in passages:
                        if not isinstance(passage, dict):
                            continue
                        if passage.get("id") == group_id or passage.get("group_id") == group_id:
                            matching_passage = passage
                            break

                if matching_passage is None:
                    raise KeyError(
                        f"Question {qid!r} in {qfile} references group_id={group_id!r}, "
                        "but no matching top-level passage exists"
                    )
                if isinstance(matching_passage, dict):
                    passage_text = matching_passage.get("text", matching_passage.get("passage"))
                else:
                    passage_text = matching_passage
                if passage_text is None:
                    raise KeyError(
                        f"Question {qid!r} in {qfile} references group_id={group_id!r}, "
                        "but the matching top-level passage has no text"
                    )
                sections.append(f"Passage:\n{render_question_value(passage_text)}")

            sections.append(f"Question:\n{render_question_value(q['stem'])}")

            options_section = render_numbered_section("Options", q.get("options"))
            if options_section is not None:
                sections.append(options_section)

            pairs = q.get("pairs")
            if isinstance(pairs, dict):
                for side in ("left", "right"):
                    pair_section = render_numbered_section(f"Pairs ({side})", pairs.get(side))
                    if pair_section is not None:
                        sections.append(pair_section)
            elif pairs is not None:
                pair_section = render_numbered_section("Pairs", pairs)
                if pair_section is not None:
                    sections.append(pair_section)

            items_section = render_numbered_section("Items", q.get("items"))
            if items_section is not None:
                sections.append(items_section)

            return QuestionContext(
                query="\n\n".join(sections),
                answer=q.get("answer"),
                explanation=q.get("explanation"),
            )
    raise KeyError(f"Question {qid!r} not found in {qfile}")
