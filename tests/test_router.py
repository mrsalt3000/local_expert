"""Тесты авто-routing: выбор книг, порог уверенности и fallback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus, Chunk, Section
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderError,
    ProviderHealth,
)
from book_expert.retrieval.router import BookRouter


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0
        self.messages: list[ChatMessage] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.messages = list(messages)
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.text, model="fake-chat")

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


def _catalog(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Retrieval Handbook",
            path=tmp_path / "one.pdf",
            content_hash="hash-1",
            description="Dense and sparse retrieval.",
            language="en",
            status=BookStatus.ready,
        ),
        [Section("book-1", "Chapter 1", 0, "Chapter 1", 0, 3)],
    )
    catalog.upsert_book(
        Book(
            id="book-2",
            title="Cooking Basics",
            path=tmp_path / "two.pdf",
            content_hash="hash-2",
            description="Recipes and techniques.",
            language="en",
            status=BookStatus.ready,
        ),
        [Section("book-2", "Soups", 0, "Soups", 0, 5)],
    )
    return catalog


def _settings(tmp_path: Path, *, threshold: float = 0.6) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        router_confidence_threshold=threshold,
        router_use_llm=True,
    )


def test_router_use_llm_defaults_to_off() -> None:
    assert Settings(_env_file=None).router_use_llm is False


def test_selects_book_above_threshold(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(
        '{"book_ids": ["book-1"], "confidence": 0.92, "reason": "retrieval topic"}'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("How does hybrid retrieval work?")

    assert route.book_ids == ("book-1",)
    assert route.confidence == 0.92
    assert route.fallback is False
    prompt = provider.messages[1].content
    assert "Retrieval Handbook" in prompt
    assert "Chapter 1" in prompt
    assert "How does hybrid retrieval work?" in prompt
    assert "Dense and sparse retrieval." in prompt


def test_router_use_llm_disabled_returns_candidates(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider('{"book_ids": ["book-1"], "confidence": 0.9}')
    settings = _settings(tmp_path)
    settings.router_use_llm = False
    router = BookRouter(settings, provider, catalog)

    route = router.route("How does hybrid retrieval work?")

    assert provider.calls == 0
    assert set(route.book_ids) == {"book-1"}
    assert route.fallback is False
    assert "router_use_llm=false" in route.reason


def test_low_confidence_falls_back_to_library(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(
        '{"book_ids": ["book-1"], "confidence": 0.2, "reason": "unsure"}'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("ambiguous question")

    assert set(route.book_ids) == {"book-1", "book-2"}
    assert route.fallback is True
    assert route.confidence == 0.2


def test_empty_selection_falls_back(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider('{"book_ids": [], "confidence": 0.9, "reason": "none"}')
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert set(route.book_ids) == {"book-1", "book-2"}
    assert route.fallback is True


def test_unknown_ids_fall_back(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(
        '{"book_ids": ["missing"], "confidence": 0.95, "reason": "hallucinated"}'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert set(route.book_ids) == {"book-1", "book-2"}
    assert route.fallback is True


def test_invalid_json_falls_back(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider("not json at all")
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert route.fallback is True
    assert "invalid router response" in route.reason


def test_provider_error_falls_back(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(error=ProviderError("server down"))
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert route.fallback is True
    assert "router unavailable" in route.reason


def test_router_prefilters_large_catalog_before_llm(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "large.sqlite3")
    catalog.initialize()
    for index in range(30):
        catalog.upsert_book(
            Book(
                id=f"book-{index}",
                title=f"Unrelated Topic {index}",
                path=tmp_path / f"{index}.pdf",
                content_hash=f"hash-{index}",
                status=BookStatus.ready,
            )
        )
    catalog.upsert_book(
        Book(
            id="sleep",
            title="Sleep Science",
            path=tmp_path / "sleep.pdf",
            content_hash="hash-sleep",
            status=BookStatus.ready,
        )
    )
    provider = FakeProvider(
        '{"book_ids": ["sleep"], "confidence": 0.95, "reason": "sleep topic"}'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("How can I improve deep sleep?")

    assert route.book_ids == ("sleep",)
    prompt = provider.messages[1].content
    assert "Sleep Science" in prompt
    assert "Unrelated Topic 29" not in prompt


def test_router_keeps_lexical_candidate_before_title_matches(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "lexical.sqlite3")
    catalog.initialize()
    for index in range(30):
        catalog.upsert_book(
            Book(
                id=f"model-{index}",
                title=f"Model Business {index}",
                path=tmp_path / f"model-{index}.pdf",
                content_hash=f"model-hash-{index}",
                status=BookStatus.ready,
            )
        )
    catalog.upsert_book(
        Book(
            id="deploy",
            title="ML Deployment Handbook",
            path=tmp_path / "deploy.pdf",
            content_hash="deploy-hash",
            status=BookStatus.ready,
        )
    )
    catalog.replace_chunks(
        "deploy",
        [
            Chunk(
                id="deploy-chunk",
                book_id="deploy",
                text="Deployment pipelines for ML models in production.",
                pdf_page_index=0,
                ordinal=0,
            )
        ],
    )
    provider = FakeProvider(
        '{"book_ids": ["deploy"], "confidence": 0.95, "reason": "deployment"}'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("deployment pipelines for ML models in production")

    assert route.book_ids == ("deploy",)
    assert "ML Deployment Handbook" in provider.messages[1].content


def test_router_loads_profiles_for_any_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    settings.book_profiles_path.write_text(
        json.dumps(
            {
                "model": "fake-embed",
                "dim": 2,
                "profiles": [
                    {"book_id": "book-1", "title": "Book", "vector": [1.0, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    catalog = _catalog(tmp_path)
    provider = FakeProvider()

    provider.name = "ollama"
    assert BookRouter(settings, provider, catalog).profiles is not None

    provider.name = "deepseek"
    assert BookRouter(settings, provider, catalog).profiles is not None


class _CandidateCatalog:
    def search_lexical(self, query: str, **kwargs: object) -> list[object]:
        return [SimpleNamespace(book_id="book-lex")]


class _FakeProfiles:
    def search(self, vector: object, limit: int = 8) -> list[str]:
        return ["book-prof"]


class _EmbedProvider:
    name = "ollama"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=((1.0, 0.0),), model="fake-embed", dim=2)


def test_candidate_order_is_lexical_first_and_provider_independent() -> None:
    from book_expert.retrieval.router import _candidate_books

    books = [
        Book(
            id="book-lex",
            title="Zeta",
            path=Path("/tmp/lex.pdf"),
            content_hash="h1",
            status=BookStatus.ready,
        ),
        Book(
            id="book-prof",
            title="Omega",
            path=Path("/tmp/prof.pdf"),
            content_hash="h2",
            status=BookStatus.ready,
        ),
    ]
    provider = _EmbedProvider()
    catalog = _CandidateCatalog()

    ordered = _candidate_books(
        "zeta query",
        books,
        catalog,  # type: ignore[arg-type]
        profiles=_FakeProfiles(),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    assert [book.id for book in ordered][:2] == ["book-lex", "book-prof"]


def test_empty_library_skips_llm(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "empty.sqlite3")
    catalog.initialize()
    provider = FakeProvider('{"book_ids": [], "confidence": 1.0}')
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert route.book_ids == ()
    assert route.reason == "library is empty"
    assert provider.calls == 0


def test_single_book_skips_llm(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "single.sqlite3")
    catalog.initialize()
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Retrieval Handbook",
            path=tmp_path / "one.pdf",
            content_hash="hash-1",
            status=BookStatus.ready,
        ),
        [Section("book-1", "Chapter 1", 0, "Chapter 1", 0, 3)],
    )
    provider = FakeProvider('{"book_ids": ["book-1"], "confidence": 0.2}')
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("any question")

    assert route.book_ids == ("book-1",)
    assert route.confidence == 1.0
    assert route.fallback is False
    assert "single ready book" in route.reason
    assert provider.calls == 0


def test_empty_query_skips_llm(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider('{"book_ids": ["book-1"], "confidence": 1.0}')
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("   ")

    assert route.fallback is True
    assert route.reason == "empty query"
    assert provider.calls == 0


def test_fenced_json_and_confidence_clamped(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(
        '```json\n{"book_ids": ["book-2"], "confidence": 1.7, "reason": "ok"}\n```'
    )
    router = BookRouter(_settings(tmp_path), provider, catalog)

    route = router.route("question")

    assert route.book_ids == ("book-2",)
    assert route.confidence == 1.0
    assert route.fallback is False


def test_router_writes_trace_with_redacted_query(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    provider = FakeProvider(
        '{"book_ids": ["book-1"], "confidence": 0.9, "reason": "topic"}'
    )
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    router = BookRouter(_settings(tmp_path), provider, catalog, trace=trace)

    router.route("secret routing question")

    record = json.loads(trace.path.read_text(encoding="utf-8"))
    assert record["event"] == "router_decision"
    assert record["book_ids"] == ["book-1"]
    assert record["fallback"] is False
    assert record["query"] == "<23 chars>"
    assert "secret routing question" not in trace.path.read_text(encoding="utf-8")
