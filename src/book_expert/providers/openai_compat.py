"""Адаптер OpenAI-compatible API: /chat/completions, /embeddings и /models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from book_expert.config import Settings
from book_expert.providers.base import (
    CHAT_TIMEOUT_S,
    HEALTH_TIMEOUT_S,
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderError,
    ProviderHealth,
    ProviderUnavailableError,
    fetch_json,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
HINT = "Check OPENAI_BASE_URL and network access."


class OpenAICompatibleProvider:
    """Провайдер любого OpenAI-compatible сервера."""

    name = "openai"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.chat_model = settings.openai_chat_model
        self.embedding_model = settings.openai_embedding_model
        self._base_url = (settings.openai_base_url or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = settings.openai_api_key
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _headers(self) -> dict[str, str]:
        if self._api_key is None or not self._api_key.get_secret_value():
            raise ProviderUnavailableError("OPENAI_API_KEY is not configured")
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.chat_model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        payload = fetch_json(
            self._client,
            "POST",
            self._url("/chat/completions"),
            provider=self.name,
            hint=HINT,
            headers=self._headers(),
            json=body,
            timeout=CHAT_TIMEOUT_S,
        )
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("openai: ответ не содержит choices[0].message.content")
        usage = payload.get("usage")
        return ChatResult(
            text=content,
            model=str(payload.get("model") or self.chat_model),
            usage=dict(usage) if isinstance(usage, dict) else {},
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        inputs = list(texts)
        if not inputs:
            raise ProviderError("openai: список текстов для embedding пуст")
        payload = fetch_json(
            self._client,
            "POST",
            self._url("/embeddings"),
            provider=self.name,
            hint=HINT,
            headers=self._headers(),
            json={"model": self.embedding_model, "input": inputs},
            timeout=CHAT_TIMEOUT_S,
        )
        raw = payload.get("data")
        if not isinstance(raw, list):
            raise ProviderError("openai: ответ не содержит data")
        if len(raw) != len(inputs):
            raise ProviderError(
                f"openai: ожидалось {len(inputs)} векторов, получено {len(raw)}"
            )
        entries: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("embedding"), list
            ):
                raise ProviderError("openai: некорректный элемент data")
            entries.append(entry)
        if all(isinstance(entry.get("index"), int) for entry in entries):
            entries.sort(key=lambda entry: entry["index"])
        vectors = tuple(
            tuple(float(value) for value in entry["embedding"]) for entry in entries
        )
        dim = len(vectors[0])
        if dim == 0:
            raise ProviderError("openai: получены векторы нулевой размерности")
        if any(len(vector) != dim for vector in vectors):
            raise ProviderError("openai: векторы разной размерности")
        return EmbeddingResult(
            vectors=vectors,
            model=str(payload.get("model") or self.embedding_model),
            dim=dim,
        )

    def health(self) -> ProviderHealth:
        try:
            payload = fetch_json(
                self._client,
                "GET",
                self._url("/models"),
                provider=self.name,
                hint=HINT,
                headers=self._headers(),
                timeout=HEALTH_TIMEOUT_S,
            )
        except ProviderUnavailableError:
            raise
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc))
        entries = payload.get("data")
        models: list[str] = []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    models.append(entry["id"])
        return ProviderHealth(
            ok=True,
            detail=f"openai-compatible reachable at {self._base_url}",
            models=tuple(models),
        )
