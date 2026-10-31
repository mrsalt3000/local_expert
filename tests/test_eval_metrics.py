"""Тесты eval-метрик: recall@k, MRR и citation precision/recall."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from book_expert.eval.metrics import (
    EvalCase,
    load_cases,
    run_answer_eval,
    run_retrieval_eval,
)
from book_expert.models import Answer, Citation, SearchHit
from book_expert.retrieval.hybrid import SearchOutcome

FIXTURE = Path(__file__).parent / "fixtures" / "eval_cases.synthetic.json"


class FakeSearcher:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping
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
        ids = self.mapping.get(query, [])[: limit or len(self.mapping.get(query, []))]
        hits = tuple(
            SearchHit(
                chunk_id=chunk_id,
                book_id="book-1",
                text="synthetic",
                score=1.0,
                pdf_page_index=0,
            )
            for chunk_id in ids
        )
        return SearchOutcome(hits=hits)


class FakeAnswerer:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping
        self.calls: list[dict[str, Any]] = []

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = False,
        limit: int | None = None,
        dense: bool = True,
    ) -> Answer:
        self.calls.append(
            {
                "question": question,
                "book_ids": book_ids,
                "auto_route": auto_route,
                "limit": limit,
            }
        )
        citations = tuple(
            Citation(
                book_id="book-1",
                book_title="Book One",
                pdf_page_index=0,
                chunk_id=chunk_id,
            )
            for chunk_id in self.mapping.get(question, [])
        )
        return Answer(text="answer", citations=citations)


def _cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase("q1", ("c1", "c2"), ("book-1",)),
        EvalCase("q2", ("c4",), ("book-2",)),
        EvalCase("q3", ("c6",)),
    )


def test_retrieval_metrics() -> None:
    searcher = FakeSearcher(
        {
            "q1": ["c1", "c3", "c2"],
            "q2": ["c5", "c4"],
            "q3": ["c7"],
        }
    )
    report = run_retrieval_eval(searcher, _cases(), k=5)

    assert report.recall_at_k == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx(0.5)
    assert report.cases[0].recall == 1.0
    assert report.cases[0].reciprocal_rank == 1.0
    assert report.cases[1].reciprocal_rank == 0.5
    assert report.cases[2].recall == 0.0
    assert searcher.calls[0]["book_ids"] == ["book-1"]
    assert searcher.calls[1]["book_ids"] == ["book-2"]
    assert searcher.calls[2]["book_ids"] is None


def test_retrieval_k_limits_hits() -> None:
    searcher = FakeSearcher({"q1": ["c1", "c3", "c2"]})
    report = run_retrieval_eval(searcher, [EvalCase("q1", ("c1", "c2"))], k=2)
    assert report.cases[0].retrieved_ids == ("c1", "c3")
    assert report.cases[0].recall == 0.5


def test_answer_metrics() -> None:
    answerer = FakeAnswerer({"q1": ["c1", "c3"], "q2": ["c4"], "q3": []})
    report = run_answer_eval(answerer, _cases())

    assert report.citation_precision == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert report.citation_recall == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert answerer.calls[0]["auto_route"] is False
    assert answerer.calls[0]["limit"] == 5


def test_load_cases_from_synthetic_fixture() -> None:
    cases = load_cases(FIXTURE)
    assert len(cases) == 3
    assert cases[0].expected_chunk_ids == ("synthetic-c1",)
    assert cases[1].expected_book_ids == ("synthetic-book",)


def test_load_cases_rejects_invalid(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"question": "q", "expected_chunk_ids": []}]))
    with pytest.raises(ValueError, match="expected_chunk_ids"):
        load_cases(path)
