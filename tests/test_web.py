"""Тесты веб-чата: API, ошибки и статика."""

from __future__ import annotations

import json
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
        self.super_answer = Answer(
            text="Супер ответ [1].",
            citations=self.rag_answer.citations,
            confidence=0.8,
            book_ids=("b1",),
            route_reason="single book",
            plan=("запрос 1", "запрос 2", "доп"),
            rounds=1,
            max_rounds=5,
        )
        self.advanced_answer = Answer(
            text="Развернутый ответ [1].",
            citations=self.rag_answer.citations,
            confidence=0.8,
            book_ids=("b1",),
            route_reason="single book",
            plan=("запрос 1", "запрос 2"),
        )
        self.agent_text = "Агентный ответ"
        self.rag_error: Exception | None = None
        self.advanced_error: Exception | None = None
        self.agent_error: Exception | None = None
        self.books_error: Exception | None = None
        self.rag_calls: list[tuple[str, Sequence[str] | None]] = []
        self.rag_history = None
        self.advanced_calls: list[tuple[str, Sequence[str] | None]] = []
        self.super_calls: list[tuple[str, Sequence[str] | None]] = []
        self.agent_calls: list[tuple[str, Sequence[str] | None]] = []
        self.closed = False

    def list_books(self) -> list[Book]:
        if self.books_error is not None:
            raise self.books_error
        return list(self.books)

    def ask_rag(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history=None,  # type: ignore[no-untyped-def]
    ) -> Answer:
        self.rag_calls.append((message, book_ids))
        self.rag_history = history
        if self.rag_error is not None:
            raise self.rag_error
        return self.rag_answer

    def ask_advanced(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress=None,  # type: ignore[no-untyped-def]
        history=None,  # type: ignore[no-untyped-def]
    ) -> Answer:
        self.advanced_calls.append((message, book_ids))
        if self.advanced_error is not None:
            raise self.advanced_error
        if progress is not None:
            progress({"type": "plan", "queries": ["запрос 1", "запрос 2"], "total": 2})
            progress(
                {"type": "query_start", "index": 1, "query": "запрос 1", "total": 2}
            )
            progress(
                {
                    "type": "query_done",
                    "index": 1,
                    "query": "запрос 1",
                    "hits": 1,
                    "total": 2,
                }
            )
            progress({"type": "synthesizing", "total": 2, "hits": 1})
        return self.advanced_answer

    def ask_super(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress=None,  # type: ignore[no-untyped-def]
        history=None,  # type: ignore[no-untyped-def]
    ) -> Answer:
        self.super_calls.append((message, book_ids))
        if progress is not None:
            progress({"type": "plan", "queries": ["запрос 1", "запрос 2"], "total": 2})
            progress(
                {"type": "query_start", "index": 1, "query": "запрос 1", "total": 2}
            )
            progress(
                {
                    "type": "query_done",
                    "index": 1,
                    "query": "запрос 1",
                    "hits": 1,
                    "total": 2,
                }
            )
            progress(
                {
                    "type": "round",
                    "round": 1,
                    "max_rounds": 5,
                    "queries": ["доп"],
                    "total": 3,
                }
            )
            progress({"type": "synthesizing", "total": 3, "hits": 1})
        return self.super_answer

    def ask_agent(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history=None,  # type: ignore[no-untyped-def]
    ) -> str:
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
    assert 'data-mode="rag"' in response.text
    assert 'data-mode="super"' in response.text
    assert "Агент" in response.text


def test_index_has_searchable_book_picker(client: TestClient) -> None:
    html = client.get("/").text

    assert 'role="combobox"' in html
    assert 'aria-controls="book-options"' in html
    assert 'id="book-options"' in html
    assert 'role="listbox"' in html
    assert '<select id="book"' not in html


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
    response = client.post(
        "/api/chat", json={"message": "q", "mode": "rag", "book_ids": ["b1"]}
    )
    assert response.status_code == 200
    assert backend.rag_calls == [("q", ["b1"])]


def test_rag_refusal_has_no_citations(client: TestClient, backend: FakeBackend) -> None:
    backend.rag_answer = Answer(text="мало контекста", refused=True)
    data = client.post("/api/chat", json={"message": "q", "mode": "rag"}).json()
    assert data["refused"] is True
    assert data["citations"] == []


def test_chat_defaults_to_super(client: TestClient, backend: FakeBackend) -> None:
    response = client.post("/api/chat", json={"message": "q"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "super"
    assert backend.super_calls == [("q", None)]


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
    response = client.post("/api/chat", json={"message": "q", "mode": "rag"})
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
    response = client.post("/api/chat", json={"message": "q", "mode": "rag"})
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


def test_advanced_mode_returns_plan_and_citations(
    client: TestClient, backend: FakeBackend
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "  Сложный вопрос  ", "mode": "advanced"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "advanced"
    assert data["plan"] == ["запрос 1", "запрос 2"]
    assert data["citations"][0]["number"] == 1
    assert backend.advanced_calls == [("Сложный вопрос", None)]


def test_advanced_stream_emits_progress_and_response(
    client: TestClient, backend: FakeBackend
) -> None:
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Сложный вопрос", "mode": "advanced"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]
    assert types == [
        "plan",
        "query_start",
        "query_done",
        "synthesizing",
        "done",
    ]
    assert events[-1]["response"]["plan"] == ["запрос 1", "запрос 2"]


def test_advanced_stream_rejects_other_modes(client: TestClient) -> None:
    response = client.post("/api/chat/stream", json={"message": "q", "mode": "rag"})
    assert response.status_code == 400


def test_stream_defaults_to_super(client: TestClient, backend: FakeBackend) -> None:
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Глубокий вопрос"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["response"]["mode"] == "super"
    assert backend.super_calls == [("Глубокий вопрос", None)]


def test_advanced_stream_error_event(client: TestClient, backend: FakeBackend) -> None:
    backend.advanced_error = ProviderUnavailableError("ollama is down")
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Сложный вопрос", "mode": "advanced"},
    ) as response:
        assert response.headers["cache-control"] == "no-store"
        body = "".join(response.iter_text())
    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 503


def test_super_mode_returns_rounds_and_plan(
    client: TestClient, backend: FakeBackend
) -> None:
    response = client.post(
        "/api/chat", json={"message": "Глубокий вопрос", "mode": "super"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "super"
    assert data["rounds"] == 1
    assert data["max_rounds"] == 5
    assert data["plan"] == ["запрос 1", "запрос 2", "доп"]
    assert backend.super_calls == [("Глубокий вопрос", None)]


def test_super_stream_emits_round_event(
    client: TestClient, backend: FakeBackend
) -> None:
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Глубокий вопрос", "mode": "super"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert "round" in [event["type"] for event in events]
    assert events[-1]["type"] == "done"
    assert events[-1]["response"]["mode"] == "super"
    assert events[-1]["response"]["rounds"] == 1


def test_default_mode_is_super(client: TestClient) -> None:
    html = client.get("/").text

    assert 'data-mode="super" class="active"' in html
    assert 'mode: "super"' in html


def test_mode_buttons_are_ordered_by_complexity(client: TestClient) -> None:
    html = client.get("/").text
    modes = ["rag", "super"]
    positions = [html.index(f'data-mode="{mode}"') for mode in modes]
    assert positions == sorted(positions)
    assert 'data-mode="agent"' not in html
    assert 'data-mode="advanced"' not in html


def test_chat_history_is_forwarded(client: TestClient, backend: FakeBackend) -> None:
    response = client.post(
        "/api/chat",
        json={
            "message": "А подробнее?",
            "mode": "rag",
            "history": [
                {"role": "user", "content": "Что такое пайплайн?"},
                {"role": "assistant", "content": "Это этапы обучения."},
            ],
        },
    )

    assert response.status_code == 200
    assert backend.rag_history == [
        ("user", "Что такое пайплайн?"),
        ("assistant", "Это этапы обучения."),
    ]


def test_index_has_separate_chat_sessions(client: TestClient) -> None:
    html = client.get("/").text

    assert 'id="chat-list"' in html
    assert 'id="new-chat"' in html
    assert "book-expert-chats-v1" in html


def test_index_has_expand_buttons(client: TestClient) -> None:
    html = client.get("/").text

    assert "expand-btn" in html
    assert "function expandBlock" in html
    assert "Подробнее" in html


def test_health_reports_requests_and_latency(
    client: TestClient, backend: FakeBackend
) -> None:
    client.post("/api/chat", json={"message": "вопрос", "mode": "rag"})

    payload = client.get("/api/health").json()

    assert payload["ok"] is True
    assert payload["requests"] == 1
    assert payload["refusals"] == 0
    assert payload["errors"] == 0
    assert payload["modes"] == {"rag": 1}
    assert payload["latency_ms"]["p50"] is not None


def test_health_counts_refusals_and_errors(
    client: TestClient, backend: FakeBackend
) -> None:
    backend.rag_answer = Answer(text="мало контекста", refused=True)
    client.post("/api/chat", json={"message": "вопрос", "mode": "rag"})

    backend.rag_error = ProviderUnavailableError("down")
    client.post("/api/chat", json={"message": "вопрос", "mode": "rag"})

    payload = client.get("/api/health").json()
    assert payload["requests"] == 2
    assert payload["refusals"] == 1
    assert payload["errors"] == 1


def test_feedback_is_written_without_answer_text(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "chat_id": "chat-1",
            "message_index": 3,
            "rating": "down",
            "mode": "super",
        },
    )

    assert response.status_code == 200
    path = tmp_path / "data" / "artifacts" / "feedback.local.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["rating"] == "down"
    assert entry["message_index"] == 3
    assert entry["mode"] == "super"
    assert "answer" not in entry
    assert "text" not in entry


def test_feedback_rejects_unknown_rating(client: TestClient) -> None:
    response = client.post(
        "/api/feedback", json={"chat_id": "c", "message_index": 1, "rating": "meh"}
    )

    assert response.status_code == 422


def test_index_has_feedback_buttons(client: TestClient) -> None:
    html = client.get("/").text

    assert "feedback-btn" in html
    assert "/api/feedback" in html
    assert "sendFeedback" in html
