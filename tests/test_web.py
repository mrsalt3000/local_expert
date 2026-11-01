"""Тесты веб-чата: API, ошибки и статика."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from book_expert.agent.graph import AgentError, AgentTimeoutError
from book_expert.config import Settings
from book_expert.models import Answer, Book, BookStatus, Citation
from book_expert.providers import ProviderUnavailableError
from book_expert.web.app import create_app
from book_expert.web.backend import ready_books


def _book(
    book_id: str = "b1",
    status: BookStatus = BookStatus.ready,
    title: str = "Book One",
) -> Book:
    return Book(
        id=book_id,
        title=title,
        path=Path(f"/tmp/{book_id}.pdf"),
        content_hash=f"hash-{book_id}",
        status=status,
        page_count=10,
        chunk_count=20,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.books = [_book()]
        self.rag_answer = Answer(
            text="Ответ [1].",
            citations=(
                Citation(
                    book_id="b1",
                    book_title="Book One",
                    pdf_page_index=4,
                    chunk_id="c1",
                    printed_page_label="42",
                    number=1,
                    quote="Цитата.",
                ),
            ),
            confidence=0.9,
            book_ids=("b1",),
            route_reason="single book",
        )
        self.agent_text = "Агентный ответ"
        self.rag_error: Exception | None = None
        self.agent_error: Exception | None = None
        self.books_error: Exception | None = None
        self.rag_calls: list[tuple[str, Sequence[str] | None]] = []
        self.agent_calls: list[tuple[str, Sequence[str] | None]] = []
        self.closed = False

    def list_books(self) -> list[Book]:
        if self.books_error is not None:
            raise self.books_error
        return list(self.books)

    def ask_rag(self, message: str, book_ids: Sequence[str] | None = None) -> Answer:
        self.rag_calls.append((message, book_ids))
        if self.rag_error is not None:
            raise self.rag_error
        return self.rag_answer

    def ask_agent(self, message: str, book_ids: Sequence[str] | None = None) -> str:
        self.agent_calls.append((message, book_ids))
        if self.agent_error is not None:
            raise self.agent_error
        return self.agent_text

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def client(backend: FakeBackend, tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(workspace=tmp_path, data_dir=tmp_path / "data")
    app = create_app(settings, backend)
    with TestClient(app) as test_client:
        yield test_client
    assert backend.closed


def test_index_serves_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Book Expert" in response.text
    assert 'data-mode="agent"' in response.text


def test_books_endpoint_maps_fields(client: TestClient) -> None:
    response = client.get("/api/books")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "b1",
            "title": "Book One",
            "author": None,
            "pages": 10,
            "chunks": 20,
            "status": "ready",
            "format": "pdf",
        }
    ]


def test_ready_books_filters_unavailable_statuses() -> None:
    books = [
        _book("ready"),
        _book("pending", BookStatus.pending),
        _book("failed", BookStatus.failed),
    ]
    assert [book.id for book in ready_books(books)] == ["ready"]


def test_rag_answer_returns_citations(client: TestClient, backend: FakeBackend) -> None:
    response = client.post(
        "/api/chat", json={"message": "  Что такое X?  ", "mode": "rag"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rag"
    assert data["answer"] == "Ответ [1]."
    assert data["refused"] is False
    assert data["citations"][0]["page"] == "42"
    assert data["citations"][0]["number"] == 1
    assert data["citations"][0]["quote"] == "Цитата."
    assert data["confidence"] == pytest.approx(0.9)
    assert data["elapsed_ms"] >= 0
    assert backend.rag_calls == [("Что такое X?", None)]


def test_rag_receives_book_filter(client: TestClient, backend: FakeBackend) -> None:
    response = client.post("/api/chat", json={"message": "q", "book_ids": ["b1"]})
    assert response.status_code == 200
    assert backend.rag_calls == [("q", ["b1"])]


def test_rag_refusal_has_no_citations(client: TestClient, backend: FakeBackend) -> None:
    backend.rag_answer = Answer(text="мало контекста", refused=True)
    data = client.post("/api/chat", json={"message": "q"}).json()
    assert data["refused"] is True
    assert data["citations"] == []


def test_agent_mode_returns_text(client: TestClient, backend: FakeBackend) -> None:
    response = client.post(
        "/api/chat", json={"message": "q", "mode": "agent", "book_ids": ["b1"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "agent"
    assert data["answer"] == "Агентный ответ"
    assert data["citations"] == []
    assert data["refused"] is False
    assert backend.agent_calls == [("q", ["b1"])]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "q", "mode": "other"},
        {"message": "x" * 4001},
        {"message": "q", "book_ids": [f"b{index}" for index in range(21)]},
        {"message": "q", "book_ids": ["x" * 129]},
    ],
)
def test_invalid_requests_are_rejected(
    client: TestClient, payload: dict[str, object]
) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_provider_error_is_503(client: TestClient, backend: FakeBackend) -> None:
    backend.rag_error = ProviderUnavailableError("ollama is unavailable")
    response = client.post("/api/chat", json={"message": "q"})
    assert response.status_code == 503
    assert "ollama is unavailable" in response.json()["detail"]


def test_agent_provider_error_is_503(client: TestClient, backend: FakeBackend) -> None:
    backend.agent_error = ProviderUnavailableError("provider is unavailable")
    response = client.post("/api/chat", json={"message": "q", "mode": "agent"})
    assert response.status_code == 503
    assert "provider is unavailable" in response.json()["detail"]


def test_agent_error_is_502(client: TestClient, backend: FakeBackend) -> None:
    backend.agent_error = AgentError("agent failed: RuntimeError")
    response = client.post("/api/chat", json={"message": "q", "mode": "agent"})
    assert response.status_code == 502
    assert "agent failed" in response.json()["detail"]


def test_value_error_is_400(client: TestClient, backend: FakeBackend) -> None:
    backend.rag_error = ValueError("bad book filter")
    response = client.post("/api/chat", json={"message": "q"})
    assert response.status_code == 400
    assert "bad book filter" in response.json()["detail"]


def test_books_provider_error_is_503(client: TestClient, backend: FakeBackend) -> None:
    backend.books_error = ProviderUnavailableError("catalog is unavailable")
    response = client.get("/api/books")
    assert response.status_code == 503
    assert "catalog is unavailable" in response.json()["detail"]


def test_agent_timeout_is_504(client: TestClient, backend: FakeBackend) -> None:
    backend.agent_error = AgentTimeoutError("agent exceeded 1s")
    response = client.post("/api/chat", json={"message": "q", "mode": "agent"})
    assert response.status_code == 504
    assert "agent exceeded" in response.json()["detail"]


def test_index_renders_markdown_safely(client: TestClient) -> None:
    html = client.get("/").text

    assert "function renderMarkdown" in html
    assert "bubble.append(renderMarkdown" in html
    assert "innerHTML" not in html
    assert "if (!paragraphLines.length)" in html
    assert "const fence = line.match(/^\\s*```(.*)$/)" in html
    assert 'document.createElement("details")' in html
    assert "sources-title" in html
