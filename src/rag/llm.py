"""Thin wrappers over the chat endpoints this project talks to."""

import openai


class OpenAICompatClient:
    """Chat client for OpenAI-compatible routes: OpenAI, LMStudio, Ollama, vLLM, Metis."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        #: Optional OpenAI ``response_format`` (e.g. a strict ``json_schema``). When set,
        #: the server constrains decoding to the schema, so the reply parses without
        #: prompt-level pleading for "JSON only".
        self.response_format = response_format

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* and return the assistant reply text."""
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.response_format is not None:
            request["response_format"] = self.response_format

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class GeminiClient:
    """Gemini chat client for Metis's native Google GenAI route.

    Metis serves the Gemini family through Google's own protocol
    (``POST {base_url}/v1beta/models/{model}:generateContent``), not through its
    OpenAI-compatible route, so these models are unreachable with
    :class:`OpenAICompatClient`.

    Automatic function calling is disabled: this client passes no tools, and leaving it on
    wraps every request in a tool-calling loop that logs a line per call.
    """

    #: Levels the Gemini API accepts. Which subset a given model supports is model
    #: dependent, so the caller picks the value and the server has final say. Gemini 3.5
    #: and newer reject the older ``thinking_budget`` field outright.
    THINKING_LEVELS = ("minimal", "low", "medium", "high")

    def __init__(
        self,
        base_url: str | None,
        api_key: str,
        model: str,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        thinking_level: str | None = "minimal",
        response_schema: dict | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai.types import (
                AutomaticFunctionCallingConfig,
                GenerateContentConfig,
                HttpOptions,
                ThinkingConfig,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Gemini models need the google-genai SDK. Install it with:\n"
                "  uv pip install google-genai"
            ) from exc

        # The SDK enum is case-insensitive and accepts unknown values, so a typo would
        # only surface as a server-side 400 once per call. Reject it here instead.
        level = (thinking_level or "").strip().lower()
        if level and level not in self.THINKING_LEVELS:
            raise ValueError(
                f"Unsupported thinking level {thinking_level!r}; "
                f"use one of {', '.join(self.THINKING_LEVELS)}, or an empty value to let "
                "the model choose"
            )

        http_options = HttpOptions(base_url=base_url) if base_url else None
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self.config = GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=ThinkingConfig(thinking_level=level.upper()) if level else None,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, prompt: str) -> str:
        """Send *prompt* as a single user turn and return the reply text.

        Returns an empty string when the model produced no text, which includes a reply
        truncated by ``max_output_tokens``: thinking tokens draw from the same budget, and
        no Gemini 3.x model lets thinking be turned off entirely.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.config,
        )
        return response.text or ""
