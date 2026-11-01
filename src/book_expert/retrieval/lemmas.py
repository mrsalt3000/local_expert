"""Русская лемматизация для FTS5 и query normalization."""

from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("pymorphy3 is not installed; run `uv sync`") from exc
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=1_000_000)
def _lemma_word(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def lemmatize_text(text: str) -> str:
    return " ".join(
        _lemma_word(match.group(0).casefold()) for match in _TOKEN_RE.finditer(text)
    )


def lemmatize_tokens(text: str, *, limit: int = 32) -> list[str]:
    return [
        _lemma_word(match.group(0).casefold()) for match in _TOKEN_RE.finditer(text)
    ][:limit]


__all__ = ["lemmatize_text", "lemmatize_tokens"]
