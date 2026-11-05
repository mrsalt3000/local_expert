"""Тесты эвристики claim-типов."""

from __future__ import annotations

from book_expert.eval.claim_types import classify


def test_short_fragment_is_meta() -> None:
    assert classify("Согласно рис.")[0] == "meta"


def test_colon_heading_is_meta() -> None:
    assert classify("Раздел 15.1 перечисляет факторы:")[0] == "meta"


def test_declarative_claim_is_factual() -> None:
    assert (
        classify("Метод X снижает задержку на 20% по сравнению с baseline.")[0]
        == "factual"
    )
