"""Book-level routing profiles: один embedding на книгу."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book
from book_expert.providers import Provider

PROFILE_LIMIT = 8
SECTION_LIMIT = 12


@dataclass(frozen=True, slots=True)
class BookProfile:
    book_id: str
    title: str
    vector: tuple[float, ...]

    @property
    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.vector))


class BookProfileStore:
    def __init__(
        self,
        profiles: Sequence[BookProfile],
        *,
        model: str,
        dim: int,
    ) -> None:
        self.profiles = tuple(profiles)
        self.model = model
        self.dim = dim

    @classmethod
    def load(cls, path: Path) -> BookProfileStore | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            raw = payload.get("profiles")
            if not isinstance(raw, list):
                return None
            profiles = [
                BookProfile(
                    book_id=str(item["book_id"]),
                    title=str(item["title"]),
                    vector=tuple(float(value) for value in item["vector"]),
                )
                for item in raw
                if isinstance(item, dict)
            ]
            dim = int(payload.get("dim") or 0)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        if dim <= 0 or any(len(profile.vector) != dim for profile in profiles):
            return None
        return cls(
            profiles,
            model=str(payload.get("model") or ""),
            dim=dim,
        )

    def search(
        self, query_vector: Sequence[float], limit: int = PROFILE_LIMIT
    ) -> list[str]:
        if len(query_vector) != self.dim:
            return []
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            return []
        scored = []
        for profile in self.profiles:
            if profile.norm == 0:
                continue
            score = sum(
                left * right
                for left, right in zip(query_vector, profile.vector, strict=True)
            ) / (query_norm * profile.norm)
            scored.append((score, profile.book_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [book_id for _, book_id in scored[:limit]]


def build_profiles(
    settings: Settings,
    catalog: Catalog,
    provider: Provider,
    output: Path | None = None,
) -> int:
    books = [book for book in catalog.list_books() if book.status.value == "ready"]
    if not books:
        raise ValueError("no ready books")
    path = output or settings.book_profiles_path
    profiles: list[BookProfile] = []
    for start in range(0, len(books), 32):
        batch = books[start : start + 32]
        texts = [_profile_text(book, catalog) for book in batch]
        result = provider.embed(texts)
        if result.dim != settings.embedding_dim:
            raise ValueError(
                f"embedding dim {result.dim} != configured {settings.embedding_dim}"
            )
        for book, vector in zip(batch, result.vectors, strict=True):
            profiles.append(
                BookProfile(
                    book_id=book.id,
                    title=book.title,
                    vector=tuple(float(value) for value in vector),
                )
            )
    payload = {
        "model": provider.embedding_model,
        "dim": settings.embedding_dim,
        "profiles": [
            {
                "book_id": profile.book_id,
                "title": profile.title,
                "vector": list(profile.vector),
            }
            for profile in profiles
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return len(profiles)


def _profile_text(book: Book, catalog: Catalog) -> str:
    parts = [book.title]
    if book.author:
        parts.append(book.author)
    if book.description:
        parts.append(book.description)
    sections = [
        section.title or section.section_path
        for section in catalog.get_outline(book.id)
    ]
    if sections:
        parts.append("Разделы: " + "; ".join(sections[:SECTION_LIMIT]))
    return "\n".join(parts)


__all__ = ["BookProfile", "BookProfileStore", "build_profiles"]
