"""Тесты HTTP-API пакетного импорта книг из папки."""

from __future__ import annotations

import time
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

    def index_pdf(
        self, path: str | Path, *, title: str | None = None, force: bool = False
    ) -> Book:
        self.calls.append((Path(path).name, force))
        return Book(
            id="book-imported",
            title=title or Path(path).stem,
            path=Path(path),
            content_hash=f"hash-{Path(path).name}",
            status=BookStatus.ready,
            page_count=1,
            chunk_count=1,
        )

    def delete_book(self, book_id: str) -> bool:
        return True


@pytest.fixture()
def stack(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
    )
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    indexer = FakeIndexer()
    service = LibraryService(settings, catalog, indexer)
    app = create_app(settings, library=service)
    with TestClient(app) as client:
        yield client, settings, indexer


def _wait_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in {"done", "failed", "cancelled"}:
            return data
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_scan_import_endpoint(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, _ = stack
    source = settings.workspace / "library"
    (source / "sub").mkdir(parents=True)
    (source / "one.pdf").write_bytes(PDF_BYTES)
    (source / "sub" / "two.pdf").write_bytes(PDF_BYTES)
    (source / "notes.txt").write_text("skip", encoding="utf-8")

    response = client.post(
        "/api/library/scan",
        json={"path": str(source), "recursive": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pdf_count"] == 2
    assert data["book_count"] == 2
    assert data["to_copy"] == 2
    assert data["batch_size"] == 2
    assert data["remaining"] == 0
    assert data["non_pdf_count"] == 1


def test_scan_accepts_windows_path_shape(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = stack

    response = client.post(
        "/api/library/scan",
        json={"path": "Z:\\definitely\\missing", "recursive": False},
    )

    assert response.status_code == 400
    assert "directory not found" in response.json()["detail"]


def test_import_endpoint_copies_and_ingests(stack) -> None:  # type: ignore[no-untyped-def]
    client, settings, indexer = stack
    source = settings.workspace / "library"
    (source / "sub").mkdir(parents=True)
    (source / "one.pdf").write_bytes(PDF_BYTES)
    (source / "sub" / "two.pdf").write_bytes(PDF_BYTES + b"x")

    response = client.post(
        "/api/library/import",
        json={"path": str(source), "recursive": True, "ingest": True},
    )

    assert response.status_code == 202
    job = response.json()
    finished = _wait_job(client, job["id"])
    assert finished["status"] == "done"
    assert finished["processed"] == 2
    assert finished["copied"] == 2
    assert finished["ingested"] == 2
    assert (settings.books_dir / "one.pdf").is_file()
    assert (settings.books_dir / "two.pdf").is_file()
    assert indexer.calls == [("one.pdf", False), ("two.pdf", False)]


def test_cancel_unknown_job_is_404(stack) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = stack

    assert client.post("/api/jobs/nope/cancel").status_code == 404
