"""Two independent LLM judges scoring one shared rubric.

The primary judge scores every evaluated row; the secondary judge scores it when that call
succeeds. A secondary failure leaves the row successful with primary scores only, so it does
not turn a successful evaluation into a failed row. Luna labelled the DPO preference pairs
and was the ROPG teacher, so a Luna-only number for those arms grades a student against its
own teacher. The secondary judge is independent of both, and ``judge_agreement.csv`` is what
makes the pair reportable.

Both judges receive byte-identical system text and byte-identical user text. The only
differences are the transport (OpenAI-compatible chat versus Google's native GenAI route)
and the sampling knobs each endpoint exposes.

A judge that cannot produce a parseable, in-range score raises :class:`JudgeError`. It never
returns a sentinel, a zero, or a -1: a failure that looks like a score silently drags every
arm's mean toward the failure rate of its endpoint rather than the quality of its answers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from data.settings import GEMINI_API_KEY, GEMINI_ENDPOINT, OPENAI_API_KEY, OPENAI_BASE_URL

if TYPE_CHECKING:
    from collections.abc import Callable

#: Scored 0..4.
INTEGER_FIELDS = ("context_utility", "faithfulness", "persona_alignment", "pedagogical_quality")
#: Scored 0 or 1.
BINARY_FIELDS = ("answer_correctness",)
SCORE_FIELDS = (
    "context_utility",
    "answer_correctness",
    "faithfulness",
    "persona_alignment",
    "pedagogical_quality",
)

JUDGE_SYSTEM = (
    "You are an impartial evaluator for a Persian educational RAG system. You receive a "
    "learner profile, a question, the gold answer and its gold explanation, the passages "
    "that were retrieved for this learner, and the response a system produced.\n\n"
    "Score five dimensions.\n\n"
    "context_utility (0-4): how relevant and sufficient the retrieved passages are for "
    "answering this learner's question. Judge the passages, not the response.\n"
    "faithfulness (0-4): whether every claim in the response is supported by the retrieved "
    "passages and none contradicts them. A fluent response that goes beyond or against the "
    "passages scores low however correct it sounds.\n"
    "persona_alignment (0-4): how well the depth, vocabulary, and style of the response fit "
    "the stated learner.\n"
    "pedagogical_quality (0-4): whether the response explains the correct answer in a way "
    "that improves this learner's understanding.\n\n"
    "Anchors for those four: 0 = failed, 1 = poor, 2 = partial, 3 = good, 4 = fully "
    "satisfied.\n\n"
    "answer_correctness (0 or 1): 1 when the response agrees with the gold answer, 0 when it "
    "does not or leaves it unclear. The gold answer and gold explanation are reference "
    "material for you only; the system that produced the response never saw them.\n\n"
    "Do not award a holistic or overall score. Reply with strict JSON and nothing else: no "
    "markdown, no code fence, no commentary. Exactly these five keys, each an integer:\n"
    '{"context_utility": <int>, "answer_correctness": <int>, "faithfulness": <int>, '
    '"persona_alignment": <int>, "pedagogical_quality": <int>}'
)

# Pins the reply shape server-side for the Gemini route, mirroring
# benchmarks/compare_dpo_rewriters.py. `parse_judge_scores` still runs on the result: a
# schema fixes the keys and their types, not their ranges.
_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {field: {"type": "INTEGER"} for field in SCORE_FIELDS},
    "required": list(SCORE_FIELDS),
    "property_ordering": list(SCORE_FIELDS),
}


@dataclass(frozen=True)
class JudgeScores:
    """One judge's verdict on one response."""

    context_utility: int
    answer_correctness: int
    faithfulness: int
    persona_alignment: int
    pedagogical_quality: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff for one remote role.

    Redefined here rather than imported from ``benchmarks/compare_dpo_rewriters.py``, which
    carries the identical shape: importing that module would pull ``rl.dpo_train`` and
    ``rag.rewriter`` — and through them unsloth — into a job that only needs a dataclass.
    """

    max_attempts: int
    initial_backoff_seconds: float
    backoff_multiplier: float
    max_backoff_seconds: float

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "retry") -> RetryPolicy:
        policy = cls(
            max_attempts=int(config["max_attempts"]),
            initial_backoff_seconds=float(config["initial_backoff_seconds"]),
            backoff_multiplier=float(config["backoff_multiplier"]),
            max_backoff_seconds=float(config["max_backoff_seconds"]),
        )
        if policy.max_attempts < 1:
            raise ValueError(f"{label}.max_attempts must be at least 1")
        if policy.initial_backoff_seconds < 0 or policy.backoff_multiplier < 1:
            raise ValueError(f"{label} backoff values are invalid")
        if policy.max_backoff_seconds < policy.initial_backoff_seconds:
            raise ValueError(f"{label} maximum must not be below its initial delay")
        return policy


class JudgeError(RuntimeError):
    """A judge exhausted its retries without returning a parseable, in-range score."""


def build_judge_user_message(
    profile_rendered: str,
    query: str,
    gold_answer: str,
    gold_explanation: str,
    passages: list[tuple[str, str]],
    answer: str,
) -> str:
    """Render the judge's user turn.

    Passages arrive as ``(chunk_id, text)`` and are numbered from 1 in retrieval order, so
    the numbers match the citations the generator was told to use. Nothing is truncated:
    cutting a Persian passage at a fixed character count severs it mid-sentence and makes
    ``faithfulness`` unjudgeable for exactly the rows where grounding is hardest.
    """
    if passages:
        passages_block = "\n\n".join(
            f"[{index}] {text}" for index, (_chunk_id, text) in enumerate(passages, start=1)
        )
    else:
        passages_block = "(no passages were retrieved)"
    return (
        f"Learner profile:\n{profile_rendered}\n\n"
        f"Question:\n{query}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Gold explanation:\n{gold_explanation}\n\n"
        f"Retrieved passages:\n{passages_block}\n\n"
        f"System response:\n{answer}"
    )


def parse_judge_scores(raw: str) -> JudgeScores:
    """Parse a judge reply into :class:`JudgeScores`, or raise ``ValueError``.

    Strict on purpose, and deliberately without fence stripping. A judge that wraps its
    JSON in markdown is not following the rubric prompt, and quietly repairing its output
    here would hide that drift behind scores nobody re-reads.
    """
    if not raw or not raw.strip():
        raise ValueError("judge returned an empty reply")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"judge reply is {type(payload).__name__}, expected an object")

    expected = set(SCORE_FIELDS)
    seen = set(payload)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"judge reply keys wrong: missing={missing} extra={extra}")

    values: dict[str, int] = {}
    for field in SCORE_FIELDS:
        value = payload[field]
        # bool is an int subclass, and `true` for answer_correctness would otherwise pass
        # every range check below and silently score as 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"judge reply {field}={value!r} is not an integer")
        upper = 1 if field in BINARY_FIELDS else 4
        if not 0 <= value <= upper:
            raise ValueError(f"judge reply {field}={value!r} is outside 0..{upper}")
        values[field] = value
    return JudgeScores(**values)


def _score_with_retry(
    call: Callable[[], str], retry: RetryPolicy, judge: str
) -> tuple[JudgeScores, int]:
    """Call *call* until it yields a parseable score, then return ``(scores, attempts)``."""
    delay = retry.initial_backoff_seconds
    last_error = "no attempt was made"
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return parse_judge_scores(call()), attempt
        except Exception as exc:  # transport and parse failures retry alike
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retry.max_attempts:
                time.sleep(delay)
                delay = min(delay * retry.backoff_multiplier, retry.max_backoff_seconds)
    raise JudgeError(
        f"{judge} failed after {retry.max_attempts} attempts; last error: {last_error}"
    )


class LunaJudge:
    """Primary judge over the OpenAI-compatible route."""

    def __init__(
        self,
        model: str,
        temperature: float,
        reasoning_effort: str | None,
        max_completion_tokens: int,
        retry: RetryPolicy,
    ) -> None:
        from rag.llm import OpenAICompatClient

        self.model = model
        self.retry = retry
        # OpenAICompatClient's `max_tokens` is sent as `max_completion_tokens` (src/rag/llm.py).
        self.client = OpenAICompatClient(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

    def score(
        self,
        *,
        profile_rendered: str,
        query: str,
        gold_answer: str,
        gold_explanation: str,
        passages: list[tuple[str, str]],
        answer: str,
    ) -> tuple[JudgeScores, int]:
        user = build_judge_user_message(
            profile_rendered, query, gold_answer, gold_explanation, passages, answer
        )
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
        return _score_with_retry(lambda: self.client.chat(messages), self.retry, self.model)


class GeminiJudge:
    """Secondary judge over Metis's native Google GenAI route."""

    def __init__(
        self,
        model: str,
        temperature: float,
        thinking_level: str | None,
        max_output_tokens: int,
        retry: RetryPolicy,
    ) -> None:
        from rag.llm import GeminiClient

        self.model = model
        self.retry = retry
        self.client = GeminiClient(
            base_url=GEMINI_ENDPOINT,
            api_key=GEMINI_API_KEY,
            model=model,
            system_instruction=JUDGE_SYSTEM,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            response_schema=_GEMINI_SCHEMA,
        )

    def score(
        self,
        *,
        profile_rendered: str,
        query: str,
        gold_answer: str,
        gold_explanation: str,
        passages: list[tuple[str, str]],
        answer: str,
    ) -> tuple[JudgeScores, int]:
        user = build_judge_user_message(
            profile_rendered, query, gold_answer, gold_explanation, passages, answer
        )
        return _score_with_retry(lambda: self.client.generate(user), self.retry, self.model)
