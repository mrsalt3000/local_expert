"""Метрики retrieval и grounded answer на synthetic eval-наборе."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher

Item = TypeVar("Item", bound=Hashable)


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    expected_chunk_ids: tuple[str, ...]
    expected_book_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    question: str
    retrieved_ids: tuple[str, ...]
    expected_ids: tuple[str, ...]
    recall: float
    reciprocal_rank: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    k: int
    cases: tuple[RetrievalCaseResult, ...]
    recall_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class AnswerCaseResult:
    question: str
    cited_ids: tuple[str, ...]
    expected_ids: tuple[str, ...]
    citation_precision: float
    citation_recall: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerReport:
    cases: tuple[AnswerCaseResult, ...]
    citation_precision: float
    citation_recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "cases": [case.to_dict() for case in self.cases],
        }


def load_cases(path: str | Path) -> tuple[EvalCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("eval file must be a non-empty JSON list")
    cases: list[EvalCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        question = item.get("question")
        expected = item.get("expected_chunk_ids")
        books = item.get("expected_book_ids", [])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"case {index}: question is required")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(value, str) for value in expected)
        ):
            raise ValueError(f"case {index}: expected_chunk_ids must be non-empty")
        if not isinstance(books, list) or not all(
            isinstance(value, str) for value in books
        ):
            raise ValueError(f"case {index}: expected_book_ids must be a list")
        cases.append(EvalCase(question, tuple(expected), tuple(books)))
    return tuple(cases)


def run_retrieval_eval(
    searcher: HybridSearcher,
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    dense: bool = True,
) -> RetrievalReport:
    if k <= 0:
        raise ValueError("k must be positive")
    results: list[RetrievalCaseResult] = []
    for case in cases:
        outcome = searcher.search(
            case.question,
            book_ids=list(case.expected_book_ids) or None,
            limit=k,
            dense=dense,
        )
        retrieved = tuple(hit.chunk_id for hit in outcome.hits)
        expected = set(case.expected_chunk_ids)
        relevant = [chunk_id for chunk_id in retrieved if chunk_id in expected]
        recall = len(relevant) / len(expected) if expected else 0.0
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                reciprocal_rank = 1.0 / rank
                break
        results.append(
            RetrievalCaseResult(
                question=case.question,
                retrieved_ids=retrieved,
                expected_ids=case.expected_chunk_ids,
                recall=recall,
                reciprocal_rank=reciprocal_rank,
            )
        )
    return RetrievalReport(
        k=k,
        cases=tuple(results),
        recall_at_k=_mean(case.recall for case in results),
        mrr=_mean(case.reciprocal_rank for case in results),
    )


def run_answer_eval(
    answerer: GroundedAnswerer,
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    auto_route: bool = False,
    dense: bool = True,
) -> AnswerReport:
    results: list[AnswerCaseResult] = []
    for case in cases:
        answer = answerer.ask(
            case.question,
            book_ids=list(case.expected_book_ids) or None,
            auto_route=auto_route,
            limit=k,
            dense=dense,
        )
        cited = tuple(citation.chunk_id for citation in answer.citations)
        expected = set(case.expected_chunk_ids)
        cited_set = set(cited)
        intersection = cited_set & expected
        precision = len(intersection) / len(cited_set) if cited_set else 0.0
        recall = len(intersection) / len(expected) if expected else 0.0
        results.append(
            AnswerCaseResult(
                question=case.question,
                cited_ids=cited,
                expected_ids=case.expected_chunk_ids,
                citation_precision=precision,
                citation_recall=recall,
            )
        )
    return AnswerReport(
        cases=tuple(results),
        citation_precision=_mean(case.citation_precision for case in results),
        citation_recall=_mean(case.citation_recall for case in results),
    )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def hit_at_k(
    retrieved: Sequence[Item],
    grades: Mapping[Item, int],
    k: int,
    *,
    min_grade: int = 1,
) -> float:
    """1.0, если в top-k есть хотя бы один допустимый фрагмент.

    Для multi-gold разметки это метрика уровня кейса: неважно, какой именно
    из допустимых фрагментов найден, важно, что найден любой.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return float(
        any(grades.get(chunk_id, 0) >= min_grade for chunk_id in retrieved[:k])
    )


def recall_at_k(
    retrieved: Sequence[Item],
    grades: Mapping[Item, int],
    k: int,
    *,
    min_grade: int = 1,
) -> float:
    """Доля найденных допустимых фрагментов от всех размеченных."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {
        chunk_id
        for chunk_id, grade in grades.items()
        if grade >= min_grade and grade > 0
    }
    if not relevant:
        return 0.0
    found = {chunk_id for chunk_id in retrieved[:k] if chunk_id in relevant}
    return len(found) / len(relevant)


def map_at_k(
    retrieved: Sequence[Item],
    grades: Mapping[Item, int],
    k: int,
    *,
    min_grade: int = 1,
) -> float:
    """MAP@k по бинарной релевантности (grade >= min_grade).

    Знаменатель — ``min(число допустимых, k)``: если допустимых фрагментов
    меньше k, находить «все k» невозможно.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {
        chunk_id
        for chunk_id, grade in grades.items()
        if grade >= min_grade and grade > 0
    }
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def ndcg_at_k(
    retrieved: Sequence[Item],
    grades: Mapping[Item, int],
    k: int,
) -> float:
    """nDCG@k с graded relevance: gain = 2**grade - 1."""
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = sum(
        (2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted((grade for grade in grades.values() if grade > 0), reverse=True)
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal[:k], start=1)
    )
    return dcg / idcg if idcg else 0.0


__all__ = [
    "AnswerCaseResult",
    "AnswerReport",
    "EvalCase",
    "RetrievalCaseResult",
    "RetrievalReport",
    "hit_at_k",
    "load_cases",
    "map_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "run_answer_eval",
    "run_retrieval_eval",
]
