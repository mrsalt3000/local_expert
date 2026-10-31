"""Контракты и общие ошибки слоя провайдеров chat/embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

CHAT_TIMEOUT_S = 120.0
HEALTH_TIMEOUT_S = 15.0
BODY_LIMIT = 300


class ProviderError(RuntimeError):
    """Некорректный ответ провайдера, HTTP-ошибка или ошибка конфигурации."""


class ProviderUnavailableError(ProviderError):
    """Сервер провайдера недоступен или не сконфигурирован ключ доступа."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    dim: int


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    ok: bool
    detail: str
    models: tuple[str, ...] = ()


@runtime_checkable
class Provider(Protocol):
    """Синхронный интерфейс провайдера chat/embeddings."""

    name: str
    chat_model: str
    embedding_model: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def health(self) -> ProviderHealth: ...


def short_body(response: httpx.Response) -> str:
    """Возвращает усечённое тело ответа для сообщения об ошибке."""
    text = response.text.strip()
    return text[:BODY_LIMIT] if text else "<empty body>"


def fetch_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    provider: str,
    hint: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Выполняет один HTTP-запрос и возвращает JSON-объект ответа."""
    try:
        response = client.request(method, url, **kwargs)
        response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        message = f"{provider}: cannot reach {url}: {exc}"
        if hint:
            message = f"{message}. {hint}"
        raise ProviderUnavailableError(message) from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            f"{provider}: HTTP {exc.response.status_code} from {url}: "
            f"{short_body(exc.response)}"
        ) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider}: invalid JSON from {url}") from exc
    if not isinstance(payload, dict):
        raise ProviderError(f"{provider}: unexpected payload from {url}")
    return payload
