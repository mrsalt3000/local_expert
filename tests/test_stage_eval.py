"""Тесты per-stage eval: routing, retrieval, citations, latency, gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from book_expert.config import Settings
from book_expert.eval import stage_eval
from book_expert.eval.metrics import EvalCase
from book_expert.models import Answer, Citation, SearchHit
from book_expert.retrieval.hybrid import SearchOutcome


class FakeCatalog:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.closed = False

    def initialize(self) -> None:
        pass

    def iter_chunks(self, book_id: str | None = None) -> list[Any]:
        return self.chunks

    def list_books(self) -> list[Any]:
        return [SimpleNamespace(id="book-1", status=SimpleNamespace(value="ready"))]


class FakeProvider:
    def close(self) -> None:
        pass


class FakeSearcher:
    def __init__(self, mapping: dict[str, list[SearchHit]]) -> None:
        self.mapping = mapping
        self.queries: list[str] = []
        self.book_scopes: list[list[str] | None] = []
        self.closed = False

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int = 5,
    ) -> SearchOutcome:
        self.queries.append(query)
        self.book_scopes.append(book_ids)
        return SearchOutcome(hits=tuple(self.mapping.get(query, [])))

    def close(self) -> None:
        self.closed = True


class FakeRouter:
    def __init__(self, mapping: dict[str, list[str]], fallback: bool = False) -> None:
        self.mapping = mapping
        self.fallback = fallback
        self.profiles = None

    def route(self, query: str) -> SimpleNamespace:
        return SimpleNamespace(
            book_ids=self.mapping.get(query, []),
            fallback=self.fallback,
        )


class FakeAnswerer:
    def __init__(self, mapping: dict[str, tuple[list[Citation], bool]]) -> None:
        self.mapping = mapping

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        limit: int = 5,
    ) -> Answer:
        citations, refused = self.mapping.get(question, ([], True))
        return Answer(text="answer", citations=tuple(citations), refused=refused)


def _hit(chunk_id: str, page: int) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        book_id="book-1",
        text="synthetic",
        score=1.0,
        pdf_page_index=page,
    )


def _citation(chunk_id: str, page: int) -> Citation:
    return Citation(
        book_id="book-1",
        book_title="Book One",
        pdf_page_index=page,
        chunk_id=chunk_id,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    searcher: FakeSearcher,
    router: FakeRouter,
    answerer: FakeAnswerer,
    chunk_pages: dict[str, int],
) -> None:
    chunks = [
        SimpleNamespace(id=chunk_id, pdf_page_index=page)
        for chunk_id, page in chunk_pages.items()
    ]
    monkeypatch.setattr(stage_eval, "Catalog", lambda path: FakeCatalog(chunks))
    monkeypatch.setattr(stage_eval, "get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(stage_eval, "HybridSearcher", lambda *args, **kwargs: searcher)
    monkeypatch.setattr(stage_eval, "BookRouter", lambda *args, **kwargs: router)
    monkeypatch.setattr(stage_eval, "apply_metadata_filter", lambda books, q: books)
    monkeypatch.setattr(
        stage_eval,
        "_candidate_books",
        lambda query, books, catalog, **kwargs: books,
    )
    monkeypatch.setattr(
        stage_eval, "GroundedAnswerer", lambda *args, **kwargs: answerer
    )
    monkeypatch.setattr(
        stage_eval,
        "load_cases",
        lambda path: (
            EvalCase("q1", ("c1",), ("book-1",)),
            EvalCase("q2", ("c4",), ("book-1",)),
        ),
    )


def test_stage_eval_aggregates_each_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = FakeSearcher({"q1": [_hit("c1", 10), _hit("c9", 99)], "q2": []})
    router = FakeRouter({"q1": ["book-1"], "q2": ["book-2"]})
    answerer = FakeAnswerer(
        {
            "q1": ([_citation("c1", 10)], False),
            "q2": ([], True),
        }
    )
    _install(
        monkeypatch,
        searcher=searcher,
        router=router,
        answerer=answerer,
        chunk_pages={"c1": 10, "c4": 40},
    )

    report = stage_eval.run_stage_eval(
        Settings(query_rewrite_enabled=False),
        Path("cases.json"),
    )

    assert report["cases"] == 2
    assert report["routing"] == {
        "top1": 0.5,
        "recall_at_3": 0.5,
        "fallback": 0,
        "non_fallback": {"cases": 2, "top1": 0.5, "recall_at_3": 0.5},
    }
    assert report["retrieval"]["page_recall_at_5"] == pytest.approx(0.5)
    assert report["retrieval"]["page_mrr_at_5"] == pytest.approx(0.5)
    assert report["retrieval"]["chunk_recall_at_5"] == pytest.approx(0.5)
    assert report["answers"]["refused"] == 1
    assert report["answers"]["citation_precision"] == pytest.approx(0.5)
    assert report["answers"]["citation_recall"] == pytest.approx(0.5)
    assert report["gate"]["signal"]["fired"] == 1
    assert report["gate"]["signal"]["false_refusals"] == 0
    assert report["gate"]["oracle"]["fired"] == 1
    assert searcher.closed is True


def test_stage_eval_citation_metrics_use_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunk id после переиндексации другой, страница та же — метрика держится."""
    searcher = FakeSearcher({"q1": [_hit("c1-new", 10)], "q2": []})
    answerer = FakeAnswerer(
        {
            "q1": ([_citation("c1-new", 10)], False),
            "q2": ([], True),
        }
    )
    _install(
        monkeypatch,
        searcher=searcher,
        router=FakeRouter({}),
        answerer=answerer,
        chunk_pages={"c1": 10, "c4": 40},
    )

    report = stage_eval.run_stage_eval(
        Settings(query_rewrite_enabled=False),
        Path("cases.json"),
    )

    row = report["details"][0]
    assert row["chunk_recall@5"] == 0.0
    assert row["page_recall@5"] == 1.0
    assert row["citation_precision"] == 1.0


def test_stage_eval_reports_false_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signal-gate сработал, но ответ всё равно опирался на ожидаемую страницу."""
    searcher = FakeSearcher({"q1": []})
    answerer = FakeAnswerer({"q1": ([_citation("c1", 10)], False)})
    _install(
        monkeypatch,
        searcher=searcher,
        router=FakeRouter({}),
        answerer=answerer,
        chunk_pages={"c1": 10},
    )

    report = stage_eval.run_stage_eval(
        Settings(query_rewrite_enabled=False),
        Path("cases.json"),
    )

    assert report["gate"]["signal"]["fired"] == 2
    assert report["gate"]["signal"]["false_refusals"] == 1


def test_stage_eval_router_scope_uses_route(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = FakeSearcher({"q1": [_hit("c1", 10)], "q2": [_hit("c4", 40)]})
    _install(
        monkeypatch,
        searcher=searcher,
        router=FakeRouter({"q1": ["book-2", "book-3"], "q2": ["book-1"]}),
        answerer=FakeAnswerer({"q1": ([], True), "q2": ([], True)}),
        chunk_pages={"c1": 10, "c4": 40},
    )

    report = stage_eval.run_stage_eval(
        Settings(query_rewrite_enabled=False),
        Path("cases.json"),
        scope="router",
    )

    assert report["scope"] == "router"
    assert searcher.book_scopes == [["book-2", "book-3"], ["book-1"]]


def test_stage_eval_candidates_scope_uses_prefilter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = FakeSearcher({"q1": [], "q2": []})
    _install(
        monkeypatch,
        searcher=searcher,
        router=FakeRouter({}),
        answerer=FakeAnswerer({}),
        chunk_pages={"c1": 10, "c4": 40},
    )

    stage_eval.run_stage_eval(
        Settings(query_rewrite_enabled=False),
        Path("cases.json"),
        scope="candidates",
    )

    assert searcher.book_scopes == [["book-1"], ["book-1"]]


def test_write_report(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    stage_eval.write_report({"cases": 1}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"cases": 1}
