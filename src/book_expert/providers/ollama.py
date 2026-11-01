"""Адаптер Ollama: /api/chat, /api/embed и /api/tags."""

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

HINT = "Start Ollama with `ollama serve`."


class OllamaProvider:
    """Провайдер локального сервера Ollama."""

    name = "ollama"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.chat_model = settings.chat_model
        self.embedding_model = settings.embedding_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._think = settings.ollama_think
        self._num_ctx = settings.ollama_num_ctx
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return fetch_json(
            self._client,
            "POST",
            self._url(path),
            provider=self.name,
            hint=HINT,
            json=payload,
            timeout=CHAT_TIMEOUT_S,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": self._num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "stream": False,
                "think": self._think,
                "options": options,
            },
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("ollama: ответ не содержит message.content")
        return ChatResult(
            text=content,
            model=str(payload.get("model") or self.chat_model),
            usage=_usage(payload),
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        inputs = list(texts)
        if not inputs:
            raise ProviderError("ollama: список текстов для embedding пуст")
        payload = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": inputs},
        )
        raw = payload.get("embeddings")
        if not isinstance(raw, list):
            raise ProviderError("ollama: ответ не содержит embeddings")
        if len(raw) != len(inputs):
            raise ProviderError(
                f"ollama: ожидалось {len(inputs)} векторов, получено {len(raw)}"
            )
        vectors: list[tuple[float, ...]] = []
        for vector in raw:
            if not isinstance(vector, list):
                raise ProviderError("ollama: элемент embeddings не является списком")
            vectors.append(tuple(float(value) for value in vector))
        dim = len(vectors[0])
        if dim == 0:
            raise ProviderError("ollama: получены векторы нулевой размерности")
        if any(len(vector) != dim for vector in vectors):
            raise ProviderError("ollama: векторы разной размерности")
        return EmbeddingResult(
            vectors=tuple(vectors),
            model=str(payload.get("model") or self.embedding_model),
            dim=dim,
        )

    def health(self) -> ProviderHealth:
        try:
            payload = fetch_json(
                self._client,
                "GET",
                self._url("/api/tags"),
                provider=self.name,
                hint=HINT,
                timeout=HEALTH_TIMEOUT_S,
            )
        except ProviderUnavailableError:
            raise
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc))
        entries = payload.get("models")
        models: list[str] = []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    models.append(entry["name"])
        return ProviderHealth(
            ok=True,
            detail=f"ollama reachable at {self._base_url}",
            models=tuple(models),
        )


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for source, target in (
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "completion_tokens"),
    ):
        value = payload.get(source)
        if isinstance(value, int):
            usage[target] = value
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage
