"""CLI smoke-тесты для ingest и books без живых сервисов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
from typer.testing import CliRunner

from book_expert import cli
from book_expert.agent.audit import AuditLogger
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.indexer import IndexingError
from book_expert.models import Answer, Book, BookStatus, Citation, SearchHit
from book_expert.retrieval.hybrid import SearchOutcome

runner = CliRunner()


def _pdf(path: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Synthetic CLI text.", fontsize=12)
    document.save(path)
    document.close()
    return path


def _book() -> Book:
    return Book(
        id="book-1",
        title="Synthetic Book",
        path=Path("/tmp/synthetic.pdf"),
        content_hash="hash",
        status=BookStatus.ready,
        page_count=1,
        chunk_count=1,
    )


class FakeIndexer:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[Any, ...]] = []

    def index_pdf(
        self, path: Path, *, title: str | None = None, force: bool = False
    ) -> Book:
        self.calls.append((path, title, force))
        return _book()

    def count_points(self, book_id: str) -> int:
        return 1

    def close(self) -> None:
        self.closed = True


def test_ingest_json(monkeypatch: Any, tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "book.pdf")
    indexer = FakeIndexer()
    monkeypatch.setattr(cli, "_make_services", lambda settings: (None, None, indexer))

    result = runner.invoke(
        cli.app,
        ["ingest", str(pdf), "--title", "Synthetic Book", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["id"] == "book-1"
    assert payload["status"] == "ready"
    assert payload["points"] == 1
    assert indexer.calls == [(pdf, "Synthetic Book", False)]
    assert indexer.closed is True


def test_ingest_error_returns_exit_code(monkeypatch: Any, tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "book.pdf")

    class FailingIndexer(FakeIndexer):
        def index_pdf(
            self, path: Path, *, title: str | None = None, force: bool = False
        ) -> Book:
            raise IndexingError("no extractable text")

    indexer = FailingIndexer()
    monkeypatch.setattr(cli, "_make_services", lambda settings: (None, None, indexer))

    result = runner.invoke(cli.app, ["ingest", str(pdf), "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "no extractable text"
    assert indexer.closed is True


def test_audit_verify_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLogger(path).log("rag_search", {"query": "secret"}, ok=True)

    result = runner.invoke(cli.app, ["audit-verify", str(path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["records"] == 1


def test_books_json(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOOK_EXPERT_DATA_DIR", str(tmp_path / "data"))
    settings = Settings()
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    catalog.upsert_book(_book())

    result = runner.invoke(cli.app, ["books", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["title"] == "Synthetic Book"
    assert payload[0]["chunks"] == 1


def test_documented_commands_are_registered() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "ingest",
        "books",
        "search",
        "ask",
        "chat",
        "mcp",
        "doctor",
        "eval",
    ):
        assert name in result.stdout


class FakeSearcher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int | None = None,
        dense: bool = True,
        lexical: bool = True,
    ) -> SearchOutcome:
        self.calls.append({"query": query, "book_ids": book_ids, "limit": limit})
        return SearchOutcome(
            hits=(
                SearchHit(
                    chunk_id="c1",
                    book_id="book-1",
                    book_title="Synthetic Book",
                    text="Relevant fragment.",
                    score=0.5,
                    pdf_page_index=1,
                    printed_page_label="2",
                ),
            ),
            sources=("dense", "lexical"),
        )


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = True,
        limit: int | None = None,
        dense: bool = True,
        relevance_filter: bool = False,
    ) -> Answer:
        self.calls.append(
            {
                "question": question,
                "book_ids": book_ids,
                "auto_route": auto_route,
                "relevance_filter": relevance_filter,
            }
        )
        return Answer(
            text="Grounded answer [1].",
            citations=(
                Citation(
                    book_id="book-1",
                    book_title="Synthetic Book",
                    pdf_page_index=1,
                    chunk_id="c1",
                    printed_page_label="2",
                ),
            ),
            confidence=0.9,
            book_ids=("book-1",),
        )


class FakeServices:
    def __init__(self) -> None:
        self.searcher = FakeSearcher()
        self.answerer = FakeAnswerer()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_search_json(monkeypatch: Any) -> None:
    services = FakeServices()
    monkeypatch.setattr(cli, "_make_retrieval", lambda settings: services)

    result = runner.invoke(
        cli.app,
        [
            "search",
            "retrieval",
            "--book",
            "book-1",
            "--limit",
            "3",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["sources"] == ["dense", "lexical"]
    assert payload["hits"][0]["chunk_id"] == "c1"
    assert services.searcher.calls == [
        {"query": "retrieval", "book_ids": ["book-1"], "limit": 3}
    ]
    assert services.closed is True


def test_ask_json_no_route(monkeypatch: Any) -> None:
    services = FakeServices()
    monkeypatch.setattr(cli, "_make_retrieval", lambda settings: services)

    result = runner.invoke(
        cli.app,
        ["ask", "What?", "--book", "book-1", "--no-route", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["answer"] == "Grounded answer [1]."
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert services.answerer.calls == [
        {
            "question": "What?",
            "book_ids": ["book-1"],
            "auto_route": False,
            "relevance_filter": True,
        }
    ]
    assert services.closed is True


class LexicalSearcher:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int | None = None,
        dense: bool = True,
        lexical: bool = True,
    ) -> SearchOutcome:
        hits = self.catalog.search_lexical(
            query,
            book_ids=book_ids,
            limit=limit or 8,
        )
        return SearchOutcome(hits=tuple(hits), sources=("lexical",))


class LexicalAnswerer:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = False,
        limit: int | None = None,
        dense: bool = True,
    ) -> Answer:
        hits = self.catalog.search_lexical(
            question, book_ids=book_ids, limit=limit or 5
        )
        citations = tuple(
            Citation(
                book_id=hit.book_id,
                book_title=hit.book_title or hit.book_id,
                pdf_page_index=hit.pdf_page_index,
                chunk_id=hit.chunk_id,
            )
            for hit in hits
        )
        return Answer(text="answer", citations=citations)


class SyntheticServices:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.searcher = LexicalSearcher(catalog)
        self.answerer = LexicalAnswerer(catalog)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_eval_synthetic_returns_nonzero_metrics(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    services = SyntheticServices(catalog)
    monkeypatch.setattr(cli, "_make_retrieval", lambda settings: services)

    result = runner.invoke(cli.app, ["eval", "--synthetic", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["retrieval"]["recall_at_k"] > 0
    assert payload["retrieval"]["mrr"] > 0
    assert payload["answers"]["citation_recall"] > 0
    assert services.closed is True
    assert catalog.get_book("synthetic-book") is None
