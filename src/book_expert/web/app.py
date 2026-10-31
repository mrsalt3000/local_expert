"""FastAPI-приложение локального веб-чата."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from book_expert.agent.graph import AgentError, AgentTimeoutError
from book_expert.config import Settings
from book_expert.models import Answer, Book
from book_expert.providers import ProviderError, ProviderUnavailableError
from book_expert.web.backend import BookChatBackend
from book_expert.web.library import (
    LibraryConflictError,
    LibraryEntry,
    LibraryError,
    LibraryJob,
    LibraryNotFoundError,
    LibraryService,
    LibraryValidationError,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_MESSAGE_CHARS = 4000
MAX_BOOK_FILTERS = 20
MAX_BOOK_ID_CHARS = 128
MAX_FILE_NAME_CHARS = 255


class ChatBackend(Protocol):
    """Минимальный контракт бэкенда; в тестах подменяется fake-объектом."""

    def list_books(self) -> list[Book]: ...

    def ask_rag(
        self, message: str, book_ids: Sequence[str] | None = None
    ) -> Answer: ...

    def ask_agent(self, message: str, book_ids: Sequence[str] | None = None) -> str: ...

    def close(self) -> None: ...


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    mode: Literal["rag", "agent"] = "rag"
    book_ids: list[Annotated[str, Field(max_length=MAX_BOOK_ID_CHARS)]] = Field(
        default_factory=list, max_length=MAX_BOOK_FILTERS
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class BookOut(BaseModel):
    id: str
    title: str
    author: str | None = None
    pages: int
    chunks: int
    status: str


class CitationOut(BaseModel):
    book_id: str
    book_title: str
    page: str | int
    chunk_id: str
    quote: str | None = None


class ChatResponse(BaseModel):
    mode: Literal["rag", "agent"]
    answer: str
    refused: bool = False
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float | None = None
    route_reason: str | None = None
    elapsed_ms: int


class LibraryBookOut(BaseModel):
    key: str
    name: str
    title: str
    size_bytes: int | None = None
    file_exists: bool
    in_rag: bool
    status: str
    book_id: str | None = None
    pages: int = 0
    chunks: int = 0
    error: str | None = None


class LibraryJobOut(BaseModel):
    id: str
    kind: Literal["ingest", "reindex"]
    name: str
    status: Literal["queued", "running", "done", "failed"]
    book_id: str | None = None
    error: str | None = None


class LibraryUploadOut(BaseModel):
    entry: LibraryBookOut
    job: LibraryJobOut | None = None
    note: str | None = None


class LibraryIngestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_FILE_NAME_CHARS)
    force: bool = False


class LibraryRemoveRequest(BaseModel):
    book_id: str = Field(min_length=1, max_length=MAX_BOOK_ID_CHARS)


class LibraryDeleteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_FILE_NAME_CHARS)


class OkResponse(BaseModel):
    ok: bool = True


def create_app(
    settings: Settings | None = None,
    backend: ChatBackend | None = None,
    library: LibraryService | None = None,
) -> FastAPI:
    """Создаёт приложение; backend и library можно подменить в тестах."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        for name in ("backend", "library"):
            active = getattr(app.state, name, None)
            if active is not None:
                active.close()

    app = FastAPI(
        title="Book Expert",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings or Settings()
    if backend is not None:
        app.state.backend = backend
    if library is not None:
        app.state.library = library
    backend_lock = Lock()
    library_lock = Lock()

    def get_backend() -> ChatBackend:
        active = getattr(app.state, "backend", None)
        if active is None:
            with backend_lock:
                active = getattr(app.state, "backend", None)
                if active is None:
                    try:
                        active = BookChatBackend(app.state.settings)
                    except ValueError as exc:
                        raise ProviderUnavailableError(
                            f"backend is not configured: {exc}"
                        ) from exc
                    app.state.backend = active
        return active

    def get_library() -> LibraryService:
        active = getattr(app.state, "library", None)
        if active is None:
            with library_lock:
                active = getattr(app.state, "library", None)
                if active is None:
                    try:
                        active = LibraryService(app.state.settings)
                    except ValueError as exc:
                        raise ProviderUnavailableError(
                            f"library is not configured: {exc}"
                        ) from exc
                    app.state.library = active
        return active

    def raise_library_error(exc: Exception) -> None:
        if isinstance(exc, LibraryNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, LibraryConflictError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, LibraryValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if isinstance(exc, LibraryError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, ProviderError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, OSError):
            raise HTTPException(status_code=503, detail="local storage error") from exc
        raise exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/books", response_model=list[BookOut])
    def books() -> list[BookOut]:
        try:
            catalog_books = get_backend().list_books()
        except ProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return [_book_out(book) for book in catalog_books]

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        book_ids = payload.book_ids or None
        started = perf_counter()
        try:
            active = get_backend()
            if payload.mode == "rag":
                answer = active.ask_rag(payload.message, book_ids)
                response = _rag_response(answer)
            else:
                text = active.ask_agent(payload.message, book_ids)
                response = ChatResponse(mode="agent", answer=text, elapsed_ms=0)
        except ProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AgentTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except AgentError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response.elapsed_ms = int((perf_counter() - started) * 1000)
        return response

    @app.get("/api/library", response_model=list[LibraryBookOut])
    def library_books() -> list[LibraryBookOut]:
        try:
            entries = get_library().entries()
        except Exception as exc:
            raise_library_error(exc)
        return [_library_out(entry) for entry in entries]

    @app.post(
        "/api/library/upload",
        response_model=LibraryUploadOut,
        status_code=201,
    )
    def library_upload(
        file: UploadFile = File(...),
        ingest: bool = Form(False),
    ) -> LibraryUploadOut:
        if not file.filename:
            raise HTTPException(status_code=400, detail="file name is required")
        try:
            entry, job, note = get_library().save_upload(
                file.filename, file.file, ingest=ingest
            )
        except Exception as exc:
            raise_library_error(exc)
        finally:
            file.file.close()
        return LibraryUploadOut(
            entry=_library_out(entry),
            job=_job_out(job),
            note=note,
        )

    @app.post(
        "/api/library/ingest",
        response_model=LibraryJobOut,
        status_code=202,
    )
    def library_ingest(payload: LibraryIngestRequest) -> LibraryJobOut:
        try:
            job = get_library().start_ingest(payload.name, force=payload.force)
        except Exception as exc:
            raise_library_error(exc)
        return _job_out(job)

    @app.post("/api/library/remove-from-rag", response_model=OkResponse)
    def library_remove(payload: LibraryRemoveRequest) -> OkResponse:
        try:
            get_library().remove_from_rag(payload.book_id)
        except Exception as exc:
            raise_library_error(exc)
        return OkResponse()

    @app.post("/api/library/delete-file", response_model=OkResponse)
    def library_delete(payload: LibraryDeleteRequest) -> OkResponse:
        try:
            get_library().delete_file(payload.name)
        except Exception as exc:
            raise_library_error(exc)
        return OkResponse()

    @app.get("/api/jobs/{job_id}", response_model=LibraryJobOut)
    def library_job(job_id: str) -> LibraryJobOut:
        try:
            job = get_library().get_job(job_id)
        except Exception as exc:
            raise_library_error(exc)
        return _job_out(job)

    return app


def _library_out(entry: LibraryEntry) -> LibraryBookOut:
    return LibraryBookOut(
        key=entry.key,
        name=entry.name,
        title=entry.title,
        size_bytes=entry.size_bytes,
        file_exists=entry.file_exists,
        in_rag=entry.in_rag,
        status=entry.status,
        book_id=entry.book_id,
        pages=entry.pages,
        chunks=entry.chunks,
        error=entry.error,
    )


def _job_out(job: LibraryJob | None) -> LibraryJobOut | None:
    if job is None:
        return None
    return LibraryJobOut(
        id=job.id,
        kind=job.kind,
        name=job.name,
        status=job.status,
        book_id=job.book_id,
        error=job.error,
    )


def _book_out(book: Book) -> BookOut:
    return BookOut(
        id=book.id,
        title=book.title,
        author=book.author,
        pages=book.page_count,
        chunks=book.chunk_count,
        status=book.status.value,
    )


def _rag_response(answer: Answer) -> ChatResponse:
    return ChatResponse(
        mode="rag",
        answer=answer.text,
        refused=answer.refused,
        citations=[
            CitationOut(
                book_id=citation.book_id,
                book_title=citation.book_title,
                page=citation.printed_page_label or citation.pdf_page_index,
                chunk_id=citation.chunk_id,
                quote=citation.quote,
            )
            for citation in answer.citations
        ],
        confidence=answer.confidence,
        route_reason=answer.route_reason,
        elapsed_ms=0,
    )


__all__ = ["ChatBackend", "ChatRequest", "ChatResponse", "create_app"]
