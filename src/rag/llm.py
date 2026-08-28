"""Thin wrapper over any OpenAI-compatible chat-completions endpoint."""

import openai


class OpenAICompatClient:
    """Chat client that works with OpenAI, Gemini, LMStudio, Ollama, vLLM, etc."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

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

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""
