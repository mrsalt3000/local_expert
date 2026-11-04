"""Тесты HTTP-API библиотеки."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus
from book_expert.web.app import create_app
from book_expert.web.library import LibraryService

PDF_BYTES = b"%PDF-1.4\n% synthetic test pdf\n"


class FakeIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.deleted: list[str] = []

    def index_pdf(
        self, path: str | Path, *, title: str | None = None, force: bool = False
    ) -> Book:
        self.calls.append((Path(path).name, force))
        return Book(
            id="book-new",
            title=title or Path(path).stem,
            path=Path(path),
            content_hash="hash-new",
            status=BookStatus.ready,
            page_count=1,
            chunk_count=2,
        )

    def delete_book(self, book_id: str) -> bool:
        self.deleted.append(book_id)
        return True


@pytest.fixture()
def stack(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        max_upload_mb=1,
    )
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    indexer = FakeIndexer()
    service = LibraryService(settings, catalog, indexer)
    app = create_app(settings, library=service)
    with TestClient(app) as client:
        yield client, settings, catalog, indexer


def _wait_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in {"done", "failed"}:
            return data
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_library_lists_files_and_rag_entries(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, catalog, _ = stack
    (settings.books_dir / "new.pdf").write_bytes(PDF_BYTES)
    indexed_path = settings.books_dir / "book.pdf"
    indexed_path.write_bytes(PDF_BYTES)
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Indexed",
            path=indexed_path,
            content_hash="hash-1",
            status=BookStatus.ready,
            page_count=5,
            chunk_count=9,
        )
    )

    response = client.get("/api/library")

    assert response.status_code == 200
    items = {item["name"]: item for item in response.json()}
    assert items["new.pdf"]["status"] == "not_indexed"
    assert items["new.pdf"]["file_exists"] is True
    assert items["book.pdf"]["in_rag"] is True
    assert items["book.pdf"]["book_id"] == "book-1"


def test_upload_saves_file(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, _, _ = stack

    response = client.post(
        "/api/library/upload",
        files={"file": ("new.pdf", PDF_BYTES, "application/pdf")},
        data={"ingest": "false"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["entry"]["name"] == "new.pdf"
    assert data["entry"]["status"] == "not_indexed"
    assert data["job"] is None
    assert (settings.books_dir / "new.pdf").is_file()


def test_upload_duplicate_is_409(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, _, _ = stack
    (settings.books_dir / "book.pdf").write_bytes(PDF_BYTES)

    response = client.post(
        "/api/library/upload",
        files={"file": ("book.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 409


def test_upload_non_pdf_is_400(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = stack

    response = client.post(
        "/api/library/upload",
        files={"file": ("book.txt", b"text", "text/plain")},
    )

    assert response.status_code == 400
    assert "only PDF" in response.json()["detail"]


def test_upload_with_ingest_returns_job(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _, indexer = stack

    response = client.post(
        "/api/library/upload",
        files={"file": ("new.pdf", PDF_BYTES, "application/pdf")},
        data={"ingest": "true"},
    )

    assert response.status_code == 201
    job = response.json()["job"]
    assert job["status"] in {"queued", "running", "done"}
    finished = _wait_job(client, job["id"])
    assert finished["status"] == "done"
    assert finished["book_id"] == "book-new"
    assert indexer.calls == [("new.pdf", False)]


def test_ingest_missing_file_is_404(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = stack

    response = client.post("/api/library/ingest", json={"name": "missing.pdf"})

    assert response.status_code == 404


def test_remove_from_rag(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, catalog, indexer = stack
    path = settings.books_dir / "book.pdf"
    path.write_bytes(PDF_BYTES)
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Indexed",
            path=path,
            content_hash="hash-1",
            status=BookStatus.ready,
        )
    )

    response = client.post("/api/library/remove-from-rag", json={"book_id": "book-1"})

    assert response.status_code == 200
    assert indexer.deleted == ["book-1"]
    assert path.is_file()


def test_delete_file_removes_file_and_index(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, catalog, indexer = stack
    path = settings.books_dir / "book.pdf"
    path.write_bytes(PDF_BYTES)
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Indexed",
            path=path,
            content_hash="hash-1",
            status=BookStatus.ready,
        )
    )

    response = client.post("/api/library/delete-file", json={"name": "book.pdf"})

    assert response.status_code == 200
    assert indexer.deleted == ["book-1"]
    assert not path.exists()


def test_unknown_job_is_404(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = stack

    assert client.get("/api/jobs/nope").status_code == 404


def test_library_exposes_topics_year_and_topic_catalog(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, catalog, _ = stack
    path = settings.books_dir / "2014 - Indexed.pdf"
    path.write_bytes(PDF_BYTES)
    catalog.upsert_book(
        Book(
            id="book-topics",
            title="Indexed",
            author="Автор",
            path=path,
            content_hash="hash-topics",
            status=BookStatus.ready,
            created_at=datetime(2026, 11, 1, 12, tzinfo=UTC),
        )
    )
    catalog.set_topics("book-topics", ["IT", "Бизнес"])

    items = {item["name"]: item for item in client.get("/api/library").json()}
    item = items["2014 - Indexed.pdf"]
    assert item["topics"] == [
        "Бизнес и предпринимательство",
        "IT, программирование, архитектура",
    ]
    assert item["published_year"] == 2014
    assert item["published_year_source"] == "filename"
    assert item["added_at"] == "2026-11-01"
    assert item["author"] == "Автор"

    topics = client.get("/api/library/topics")
    assert topics.status_code == 200
    labels = topics.json()["topics"]
    assert len(labels) == 15
    assert labels[0] == "Бизнес и предпринимательство"
