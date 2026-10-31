"""Cross-encoder reranking кандидатов после гибридной фузии."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from book_expert.config import Settings


class RerankError(RuntimeError):
    """Reranker недоступен или вернул некорректный результат."""


class Reranker(Protocol):
    """Оценка релевантности пар (query, document)."""

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...

    def close(self) -> None: ...


def reranker_cache_dir(settings: Settings) -> Path:
    """Постоянный кэш моделей внутри git-ignored data/artifacts."""
    return settings.artifacts_dir / "models"


class FastEmbedReranker:
    """Ленивая обёртка над fastembed TextCrossEncoder (ONNX, без torch)."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._threads = threads
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:  # pragma: no cover - missing optional install
                raise RerankError("fastembed is not installed; run `uv sync`") from exc
            try:
                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    threads=self._threads,
                    lazy_load=True,
                )
            except Exception as exc:
                raise RerankError(
                    f"cannot load reranker {self.model_name}: {exc}"
                ) from exc
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = [str(document) for document in documents]
        if not docs:
            return []
        model = self._load()
        try:
            scores = [float(score) for score in model.rerank(query, docs)]  # type: ignore[attr-defined]
        except Exception as exc:
            raise RerankError(f"reranker {self.model_name} failed: {exc}") from exc
        if len(scores) != len(docs):
            raise RerankError(
                f"reranker returned {len(scores)} scores for {len(docs)} documents"
            )
        return scores

    def close(self) -> None:
        self._model = None


def create_reranker(settings: Settings) -> Reranker | None:
    """Создаёт reranker, только если он включён в конфигурации."""
    if not settings.rerank_enabled:
        return None
    return FastEmbedReranker(
        settings.rerank_model,
        cache_dir=reranker_cache_dir(settings),
    )


__all__ = [
    "FastEmbedReranker",
    "RerankError",
    "Reranker",
    "create_reranker",
    "reranker_cache_dir",
]
