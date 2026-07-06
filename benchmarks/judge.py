from __future__ import annotations

import json
import warnings

import litellm

from data.settings import GEMINI_API_KEY, GEMINI_ENDPOINT

_JUDGE_SYSTEM = (
    "You are an impartial evaluator for a Persian educational RAG system. "
    "You will be given a learner persona, an exam question with multiple-choice options, "
    "the correct answer, retrieved passages, and a generated explanation. "
    "Score the generated explanation on these dimensions:\n"
    "- persona_alignment (1-5): Does the depth, vocabulary, and tone match this learner's level and goal?\n"
    "- pedagogical_quality (1-5): Does the explanation help the learner understand? "
    "Is the correct option identified and explained clearly? Are wrong options ruled out with good reasoning?\n"
    "- faithfulness (1-5): Are all claims grounded in the retrieved passages? "
    "A fluent answer that contradicts or ignores the passages scores 1.\n"
    "- overall (1-5): Your holistic judgment of the response quality for this specific learner.\n"
    "- accuracy (0 or 1): Did the generated explanation correctly identify the right answer? 1 if yes, 0 if no or unclear.\n\n"
    "Respond with ONLY valid JSON, no markdown, no explanation:\n"
    '{"persona_alignment": <int>, "pedagogical_quality": <int>, "faithfulness": <int>, "overall": <int>, "accuracy": <int>}'
)

_SENTINEL = {
    "persona_alignment": -1,
    "pedagogical_quality": -1,
    "faithfulness": -1,
    "overall": -1,
    "accuracy": -1,
}


class GeminiJudge:
    """LLM-as-judge using Gemini via litellm."""

    def __init__(self, model: str = "gemini/gemini-3.5-flash", max_retries: int = 3) -> None:
        self.model = model
        self.max_retries = max_retries
        litellm.api_key = GEMINI_API_KEY
        if GEMINI_ENDPOINT is not None:
            litellm.api_base = GEMINI_ENDPOINT

    def _truncate(self, text: str, max_chars: int = 300) -> str:
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def score(
        self,
        persona: str,
        query: str,
        rewritten_query: str,
        options: list[str],
        correct_option: str,
        hits: list,  # list of Hit
        answer: str,
    ) -> dict:
        passages_block = "\n".join(
            f"[{i}] {self._truncate(h.raw_text)}" for i, h in enumerate(hits, start=1)
        )
        options_block = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))

        user_msg = (
            f"Learner persona: {persona}\n\n"
            f"Question stem: {query}\n\n"
            f"Retrieval query used: {rewritten_query}\n\n"
            f"Options:\n{options_block}\n\n"
            f"Correct answer: {correct_option}\n\n"
            f"Retrieved passages:\n{passages_block}\n\n"
            f"Generated explanation:\n{answer}"
        )

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(model=self.model, messages=messages)
                raw = resp.choices[0].message.content
                # Strip markdown fences if present
                raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                result = json.loads(raw)
                # Validate keys exist
                for k in ("persona_alignment", "pedagogical_quality", "faithfulness", "overall", "accuracy"):
                    if k not in result:
                        result[k] = -1
                return result
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                if attempt < self.max_retries - 1:
                    continue
                warnings.warn(
                    f"GeminiJudge: failed to parse JSON after {self.max_retries} attempts: {e}",
                    stacklevel=2,
                )
                return _SENTINEL.copy()

        return _SENTINEL.copy()
