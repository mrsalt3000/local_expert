"""Единый слой провайдеров chat/embeddings."""

from __future__ import annotations

from book_expert.providers.base import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    Provider,
    ProviderError,
    ProviderHealth,
    ProviderUnavailableError,
)
from book_expert.providers.factory import get_provider

__all__ = [
    "ChatMessage",
    "ChatResult",
    "EmbeddingResult",
    "Provider",
    "ProviderError",
    "ProviderHealth",
    "ProviderUnavailableError",
    "get_provider",
]
