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


class DPORewriter:
    """Persona-conditioned rewriter using a DPO-fine-tuned Qwen3 model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        adapter_path: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 200,
    ) -> None:
        from unsloth import FastLanguageModel

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            max_seq_length=512,
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        FastLanguageModel.for_inference(model)
        self.model = model
        self.tokenizer = tokenizer

    def rewrite(self, profile_rendered: str, query: str) -> str:
        """Return a persona-conditioned rewrite of *query*."""
        messages = _build_rewrite_prompt(profile_rendered, query)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract only the assistant response after the final marker
        marker = "<|im_start|>assistant\n"
        if marker in decoded:
            decoded = decoded.rsplit(marker, 1)[-1]
        return decoded.strip()
