"""DeepSeek chat + локальные Ollama embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from book_expert.config import Settings
from book_expert.providers.base import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.providers.ollama import OllamaProvider
from book_expert.providers.openai_compat import OpenAICompatibleProvider


class DeepSeekProvider:
    """Облачный DeepSeek для chat; bge-m3 остаётся локальным."""

    name = "deepseek"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.chat_model = settings.deepseek_chat_model
        self.embedding_model = settings.embedding_model
        chat_settings = settings.model_copy(
            update={
                "openai_api_key": settings.deepseek_api_key,
                "openai_base_url": settings.deepseek_base_url,
                "openai_chat_model": settings.deepseek_chat_model,
            }
        )
        self._chat = OpenAICompatibleProvider(chat_settings, client=client)
        self._embed = OllamaProvider(settings, client=client)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        return self._chat.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._embed.embed(texts)

    def health(self) -> ProviderHealth:
        chat = self._chat.health()
        embed = self._embed.health()
        return ProviderHealth(
            ok=chat.ok and embed.ok,
            detail=f"deepseek chat: {chat.detail}; embeddings: {embed.detail}",
            models=chat.models + embed.models,
        )

    def close(self) -> None:
        self._chat.close()
        self._embed.close()


__all__ = ["DeepSeekProvider"]
