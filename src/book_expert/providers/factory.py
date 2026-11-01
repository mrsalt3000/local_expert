"""Выбор адаптера провайдера по настройкам."""

from __future__ import annotations

import httpx

from book_expert.config import Settings
from book_expert.providers.base import Provider
from book_expert.providers.deepseek import DeepSeekProvider
from book_expert.providers.ollama import OllamaProvider
from book_expert.providers.openai_compat import OpenAICompatibleProvider


def get_provider(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> Provider:
    """Создаёт провайдера, соответствующий settings.provider."""
    match settings.provider:
        case "ollama":
            return OllamaProvider(settings, client=client)
        case "openai":
            return OpenAICompatibleProvider(settings, client=client)
        case "deepseek":
            return DeepSeekProvider(settings, client=client)
        case _:
            raise ValueError(f"unknown provider: {settings.provider!r}")
