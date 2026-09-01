"""Persona-conditioned query rewriters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
_DEFAULT_GENERATION = {"max_new_tokens": 224, "do_sample": False}
_ALLOWED_GENERATION_KEYS = {
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
}


def build_rewrite_messages(profile_rendered: str, query: str) -> list[dict[str, str]]:
    """Build the shared training, local-inference, and remote-baseline prompt."""
    if not profile_rendered.strip():
        raise ValueError("profile_rendered must be nonempty")
    if not query.strip():
        raise ValueError("query must be nonempty")
    user = (
        f"Learner profile: {profile_rendered}\n\n"
        f"Original question: {query}\n\n"
        "Rewritten retrieval query:"
    )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _normalize_generation(generation: dict[str, Any] | None) -> dict[str, Any]:
    settings = {**_DEFAULT_GENERATION, **(generation or {})}
    unknown = sorted(set(settings) - _ALLOWED_GENERATION_KEYS)
    if unknown:
        raise ValueError(f"Unknown generation settings: {unknown}")
    max_new_tokens = settings.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("generation.max_new_tokens must be a positive integer")
    do_sample = settings.get("do_sample")
    if not isinstance(do_sample, bool):
        raise ValueError("generation.do_sample must be boolean")
    if not do_sample:
        settings.pop("temperature", None)
        settings.pop("top_p", None)
        settings.pop("top_k", None)
    return settings


def generate_rewrite_batch(
    model: Any,
    tokenizer: Any,
    profiles: list[str],
    queries: list[str],
    *,
    generation: dict[str, Any] | None = None,
) -> list[str]:
    """Generate response-only rewrites for a batch of profile/query pairs."""
    if not profiles or len(profiles) != len(queries):
        raise ValueError("profiles and queries must be nonempty lists of equal length")
    settings = _normalize_generation(generation)
    prompt_texts = [
        tokenizer.apply_chat_template(
            build_rewrite_messages(profile, query),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for profile, query in zip(profiles, queries, strict=True)
    ]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    inputs = inputs.to(model.device)
    padded_input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        **settings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = outputs[:, padded_input_length:]
    rewrites = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    cleaned = [rewrite.strip() for rewrite in rewrites]
    empty_indices = [index for index, rewrite in enumerate(cleaned) if not rewrite]
    if empty_indices:
        raise RuntimeError(f"Model returned empty rewrites at batch indices {empty_indices}")
    return cleaned


class PromptedRewriter:
    """Call a remote LLM for a persona-conditioned query rewrite."""

    def __init__(self, llm: OpenAICompatClient) -> None:
        self.llm = llm

    def rewrite(self, profile_rendered: str, query: str) -> str:
        messages = build_rewrite_messages(profile_rendered, query)
        rewrite = self.llm.chat(messages).strip()
        if not rewrite:
            raise RuntimeError("Remote rewriter returned an empty completion")
        return rewrite


class DPORewriter:
    """Run a Qwen3 base model or a promoted DPO-family LoRA adapter."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        adapter_path: str | None = None,
        device: str = "cuda",
        max_seq_length: int = 768,
        generation: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from unsloth import FastLanguageModel

        requested_device = torch.device(device)
        loader_kwargs: dict[str, Any] = {}
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
            device_index = requested_device.index if requested_device.index is not None else 0
            torch.cuda.set_device(device_index)
            resolved_device = torch.device("cuda", device_index)
            load_in_4bit = True
            # One whole replica on one GPU. Left to itself, Unsloth's planner spreads the
            # model over every visible GPU and pins the tied embedding to the output head's
            # device, so generation feeds token ids on one GPU into an embedding on another.
            loader_kwargs["device_map"] = {"": device_index}
        elif requested_device.type == "cpu":
            resolved_device = torch.device("cpu")
            load_in_4bit = False
        else:
            raise ValueError(f"Unsupported DPO rewriter device: {device!r}")

        normalized_generation = _normalize_generation(generation)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            **loader_kwargs,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        if resolved_device.type == "cpu":
            model = model.to(resolved_device)
        FastLanguageModel.for_inference(model)

        placements = {parameter.device for parameter in model.parameters()}
        if placements != {resolved_device}:
            raise RuntimeError(
                f"DPO rewriter must hold one replica on {resolved_device}, but its weights "
                "are spread over " + ", ".join(sorted(str(place) for place in placements))
            )
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.generation = normalized_generation
        self.model = model
        self.tokenizer = tokenizer

    def rewrite_batch(self, profiles: list[str], queries: list[str]) -> list[str]:
        return generate_rewrite_batch(
            self.model,
            self.tokenizer,
            profiles,
            queries,
            generation=self.generation,
        )

    def rewrite(self, profile_rendered: str, query: str) -> str:
        return self.rewrite_batch([profile_rendered], [query])[0]
