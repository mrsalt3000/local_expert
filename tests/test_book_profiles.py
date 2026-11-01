"""Тесты book-level routing profiles."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from book_expert.models import Book, BookStatus
from book_expert.providers import EmbeddingResult
from book_expert.retrieval.book_profiles import BookProfile, BookProfileStore
from book_expert.retrieval.router import _candidate_books


def _store() -> BookProfileStore:
    return BookProfileStore(
        [
            BookProfile("sleep", "Sleep", (1.0, 0.0)),
            BookProfile("code", "Code", (0.0, 1.0)),
        ],
        model="test",
        dim=2,
    )


def test_profile_search_orders_by_cosine() -> None:
    assert _store().search((0.9, 0.1), limit=2) == ["sleep", "code"]
    assert _store().search((0.1, 0.9), limit=1) == ["code"]


def test_profile_store_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profiles.local.json"
    path.write_text(
        json.dumps(
            {
                "model": "test",
                "dim": 2,
                "profiles": [
                    {"book_id": "a", "title": "A", "vector": [1.0, 0.0]},
                    {"book_id": "b", "title": "B", "vector": [0.0, 1.0]},
                ],
            }
        ),
        encoding="utf-8",
    )

    store = BookProfileStore.load(path)

    assert store is not None
    assert store.dim == 2
    assert store.search((1.0, 0.0), limit=1) == ["a"]


def test_profile_store_rejects_bad_dimension(tmp_path: Path) -> None:
    path = tmp_path / "profiles.local.json"
    path.write_text(
        json.dumps(
            {
                "model": "test",
                "dim": 3,
                "profiles": [
                    {"book_id": "a", "title": "A", "vector": [1.0, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert BookProfileStore.load(path) is None


def test_profile_store_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "profiles.local.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert BookProfileStore.load(path) is None


def test_candidate_books_uses_profile_store(tmp_path: Path) -> None:
    book = Book(
        id="sleep",
        title="Sleep",
        path=tmp_path / "sleep.pdf",
        content_hash="hash",
        status=BookStatus.ready,
    )
    catalog = SimpleNamespace(search_lexical=lambda *args, **kwargs: [])
    provider = SimpleNamespace(
        embedding_model="test",
        embed=lambda texts: EmbeddingResult(
            vectors=((1.0, 0.0),),
            model="test",
            dim=2,
        ),
    )
    store = BookProfileStore(
        [BookProfile("sleep", "Sleep", (1.0, 0.0))],
        model="test",
        dim=2,
    )

    candidates = _candidate_books(
        "sleep",
        [book],
        catalog,
        profiles=store,
        provider=provider,
    )

    assert [item.id for item in candidates] == ["sleep"]
