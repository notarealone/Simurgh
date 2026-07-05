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
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* and return the assistant reply text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
