"""Тесты контекстного сбора token usage (T104)."""

from __future__ import annotations

from collections.abc import Sequence

from book_expert.providers.base import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.providers.usage import (
    MeteredProvider,
    add_usage,
    finish_usage,
    merge_usage,
    start_usage,
)


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        return ChatResult(
            text="ok",
            model=self.chat_model,
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=(), model=self.embedding_model, dim=3)

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="ok")


def test_metered_provider_accumulates_usage() -> None:
    provider = MeteredProvider(FakeProvider())
    token = start_usage()
    try:
        provider.chat([ChatMessage(role="user", content="q")])
        provider.chat([ChatMessage(role="user", content="q")])
    finally:
        usage = finish_usage(token)

    assert usage == {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}


def test_usage_outside_context_is_ignored() -> None:
    provider = MeteredProvider(FakeProvider())
    provider.chat([ChatMessage(role="user", content="q")])
    token = start_usage()
    assert finish_usage(token) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_merge_usage_normalizes_provider_keys() -> None:
    usage: dict[str, int] = {}
    merge_usage(usage, {"input_tokens": 10, "output_tokens": 4})
    merge_usage(usage, {"prompt_tokens": 2, "completion_tokens": 1})

    assert usage == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def test_nested_contexts_do_not_leak() -> None:
    outer = start_usage()
    add_usage({"prompt_tokens": 1})
    inner = start_usage()
    add_usage({"prompt_tokens": 2})
    inner_usage = finish_usage(inner)
    outer_usage = finish_usage(outer)

    assert inner_usage["prompt_tokens"] == 2
    assert outer_usage["prompt_tokens"] == 1
