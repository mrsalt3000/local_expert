"""Сбор token usage по вызовам провайдера в пределах одного запроса."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from typing import Any

from book_expert.providers.base import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    Provider,
    ProviderHealth,
)

Usage = dict[str, int]

_current: ContextVar[Usage | None] = ContextVar("book_expert_usage", default=None)

_KEY_MAP = {
    "prompt_tokens": "prompt_tokens",
    "input_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "output_tokens": "completion_tokens",
}


def start_usage() -> Token[Usage | None]:
    """Начинает отдельный аккумулятор для текущего запроса."""
    return _current.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def finish_usage(token: Token[Usage | None]) -> Usage:
    """Возвращает накопленное и восстанавливает предыдущий контекст."""
    usage = dict(_current.get() or {})
    _current.reset(token)
    return usage


def add_usage(usage: Mapping[str, Any] | None) -> None:
    """Добавляет usage провайдера в активный аккумулятор, если он есть."""
    current = _current.get()
    if current is None or not usage:
        return
    merge_usage(current, usage)


def merge_usage(target: Usage, usage: Mapping[str, Any] | None) -> None:
    """Нормализует ключи OpenAI/Ollama/LangChain и суммирует токены."""
    if not usage:
        return
    for source, value in usage.items():
        key = _KEY_MAP.get(str(source))
        if (
            key is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        target[key] = target.get(key, 0) + int(value)
    target["total_tokens"] = target.get("prompt_tokens", 0) + target.get(
        "completion_tokens", 0
    )


class MeteredProvider:
    """Прозрачная обёртка провайдера, складывающая usage в контекст запроса."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.name = provider.name
        self.chat_model = provider.chat_model
        self.embedding_model = provider.embedding_model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        result = self._provider.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        add_usage(result.usage)
        return result

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._provider.embed(texts)

    def health(self) -> ProviderHealth:
        return self._provider.health()

    def close(self) -> None:
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


__all__ = [
    "MeteredProvider",
    "Usage",
    "add_usage",
    "finish_usage",
    "merge_usage",
    "start_usage",
]
