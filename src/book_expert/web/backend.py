"""Сборка retrieval- и агентных сервисов для веб-чата."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from dataclasses import replace

from book_expert.agent.audit import AuditLogger
from book_expert.agent.graph import AgentRunner
from book_expert.agent.worker import run_agent_in_process
from book_expert.agent.tools.coding import build_coding_tools
from book_expert.agent.tools.rag import build_rag_tools
from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Answer, Book, BookStatus
from book_expert.providers import get_provider
from book_expert.providers.usage import MeteredProvider, finish_usage, start_usage
from book_expert.retrieval.advanced import AdvancedAnswerer
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer


class BookChatBackend:
    """RAG- и агентный режимы поверх локального индекса."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.catalog = Catalog(settings.catalog_path)
        self.catalog.initialize()
        self.provider = MeteredProvider(get_provider(settings))
        self.searcher = HybridSearcher(settings, self.provider, self.catalog)
        self.trace = TraceRecorder.create(settings, prefix="web")
        self.router = BookRouter(
            settings, self.provider, self.catalog, trace=self.trace
        )
        self.answerer = GroundedAnswerer(
            settings,
            self.provider,
            self.searcher,
            self.router,
            trace=self.trace,
        )
        self.advanced = AdvancedAnswerer(
            settings,
            self.provider,
            self.searcher,
            self.router,
            trace=self.trace,
        )
        self.super_advanced = SuperAdvancedAnswerer(
            settings,
            self.provider,
            self.searcher,
            self.router,
            trace=self.trace,
        )
        self._agent: AgentRunner | None = None
        self._agent_lock = Lock()
        self._last_agent_usage: dict[str, int] = {}

    def list_books(self) -> list[Book]:
        return ready_books(self.catalog.list_books())

    @staticmethod
    def _metered(function: Callable[[], Answer]) -> Answer:
        token = start_usage()
        try:
            answer = function()
        finally:
            usage = finish_usage(token)
        return replace(answer, usage=usage)

    @property
    def agent_usage(self) -> dict[str, int]:
        return dict(self._last_agent_usage)

    def ask_rag(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        return self._metered(
            lambda: self.answerer.ask(
                message,
                book_ids=list(book_ids) if book_ids else None,
                auto_route=True,
                history=history,
                relevance_filter=self.settings.answer_relevance_filter,
            )
        )

    def ask_advanced(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        return self._metered(
            lambda: self.advanced.ask(
                message,
                book_ids=list(book_ids) if book_ids else None,
                auto_route=True,
                progress=progress,
                history=history,
            )
        )

    def ask_super(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        return self._metered(
            lambda: self.super_advanced.ask(
                message,
                book_ids=list(book_ids) if book_ids else None,
                auto_route=True,
                progress=progress,
                history=history,
            )
        )

    def ask_agent(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> str:
        if self.settings.agent_hard_timeout:
            worker_result = run_agent_in_process(
                self.settings,
                message,
                book_ids=book_ids or None,
                history=history,
            )
            self._last_agent_usage = dict(worker_result.get("usage") or {})
            return str(worker_result["text"])
        agent_result = self._agent_runner().run(
            message,
            book_ids=book_ids or None,
            history=history,
        )
        self._last_agent_usage = dict(agent_result.usage)
        return agent_result.text

    def close(self) -> None:
        self.searcher.close()
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    def _agent_runner(self) -> AgentRunner:
        with self._agent_lock:
            if self._agent is None:
                audit = AuditLogger.create(self.settings)
                tools = build_rag_tools(self.searcher, self.answerer, audit=audit)
                tools.extend(build_coding_tools(self.settings, audit=audit))
                self._agent = AgentRunner(self.settings, tools, trace=self.trace)
            return self._agent


def ready_books(books: Sequence[Book]) -> list[Book]:
    """Оставляет только книги, доступные для поиска."""
    return [book for book in books if book.status is BookStatus.ready]


__all__ = ["BookChatBackend", "ready_books"]
