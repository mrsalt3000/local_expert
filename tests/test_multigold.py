"""Тесты multi-gold метрик и разметки (T83)."""

from __future__ import annotations

import math

import pytest

from book_expert.eval.metrics import (
    EvalCase,
    hit_at_k,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
)
from book_expert.eval.multigold import (
    GRADE_ADMISSIBLE,
    GRADE_GOLD,
    ChunkRow,
    build_multigold,
    jaccard,
    shingles,
)


def test_hit_and_recall_use_admissible_alternatives() -> None:
    grades = {"gold": GRADE_GOLD, "alt": GRADE_ADMISSIBLE, "other": 0}

    assert hit_at_k(["alt"], grades, 5) == 1.0
    assert hit_at_k(["other"], grades, 5) == 0.0
    assert recall_at_k(["alt"], grades, 5) == 0.5
    assert recall_at_k(["gold", "alt"], grades, 5) == 1.0


def test_map_at_k_counts_alternatives_and_uses_binary_relevance() -> None:
    grades = {"a": GRADE_GOLD, "b": GRADE_ADMISSIBLE, "c": GRADE_ADMISSIBLE}

    # relevant = {a, b, c}; найден a на 2-м месте и c на 4-м.
    value = map_at_k(["x", "a", "y", "c"], grades, 4)

    assert value == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_map_at_k_denominator_is_capped_by_k() -> None:
    grades = {"a": GRADE_GOLD, "b": GRADE_ADMISSIBLE, "c": GRADE_ADMISSIBLE}

    assert map_at_k(["a"], grades, 1) == pytest.approx(1.0)


def test_ndcg_rewards_gold_above_alternative() -> None:
    grades = {"gold": GRADE_GOLD, "alt": GRADE_ADMISSIBLE}
    golden_first = ndcg_at_k(["gold", "alt"], grades, 5)
    alternative_first = ndcg_at_k(["alt", "gold"], grades, 5)

    assert golden_first == pytest.approx(1.0)
    assert alternative_first < golden_first
    assert 0.0 < alternative_first < 1.0


def test_ndcg_matches_hand_computed_value() -> None:
    grades = {"a": GRADE_GOLD, "b": GRADE_ADMISSIBLE, "c": GRADE_ADMISSIBLE}
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)

    assert ndcg_at_k(["x", "a", "y", "c"], grades, 4) == pytest.approx(dcg / idcg)


def test_ndcg_is_zero_without_graded_hits() -> None:
    assert ndcg_at_k(["x"], {"gold": GRADE_GOLD}, 5) == 0.0


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        hit_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError):
        map_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 1}, 0)


def test_build_multigold_grades_page_and_section_neighbours() -> None:
    chunks = [
        ChunkRow("a", "book-1", 1, 0, "Глава 1", "текст"),
        ChunkRow("b", "book-1", 1, 1, "Глава 1", "текст"),
        ChunkRow("c", "book-1", 1, 2, "Глава 1", "текст"),
        ChunkRow("d", "book-1", 2, 3, "Глава 1", "текст"),
        ChunkRow("e", "book-1", 2, 9, "Глава 1", "текст"),
        ChunkRow("f", "book-2", 1, 0, "Другое", "текст"),
    ]
    cases = [EvalCase("вопрос", ("a",), ("book-1",))]

    (annotation,) = build_multigold(cases, chunks, duplicates={0: ["f"]})

    assert annotation.grades["a"] == GRADE_GOLD
    assert annotation.reasons["a"] == "gold"
    # b и c — та же страница, что и gold.
    assert annotation.grades["b"] == GRADE_ADMISSIBLE
    assert annotation.grades["c"] == GRADE_ADMISSIBLE
    assert annotation.grades["f"] == GRADE_ADMISSIBLE
    assert annotation.reasons["f"] == "duplicate_text"
    # d и e — другая страница и ordinal вне окна, поэтому не размечены.
    assert "d" not in annotation.grades
    assert "e" not in annotation.grades


def test_build_multigold_section_neighbour_only_within_window() -> None:
    chunks = [
        ChunkRow("a", "book-1", 1, 0, "Глава 1", "текст"),
        ChunkRow("near", "book-1", 2, 1, "Глава 1", "текст"),
        ChunkRow("far", "book-1", 3, 5, "Глава 1", "текст"),
    ]
    cases = [EvalCase("вопрос", ("a",), ("book-1",))]

    (annotation,) = build_multigold(cases, chunks)

    assert annotation.grades["near"] == GRADE_ADMISSIBLE
    assert annotation.reasons["near"] == "section_neighbor"
    assert "far" not in annotation.grades


def test_unrelated_same_page_chunk_is_not_admissible() -> None:
    chunks = [
        ChunkRow("a", "book-1", 1, 0, "Глава 1", "альфа бета гамма дельта"),
        ChunkRow("b", "book-1", 1, 1, "Глава 1", "совсем другой абзац про иное"),
    ]
    cases = [EvalCase("вопрос", ("a",), ("book-1",))]

    (annotation,) = build_multigold(cases, chunks)

    assert annotation.grades == {"a": GRADE_GOLD}


def test_similar_same_page_chunk_is_admissible() -> None:
    chunks = [
        ChunkRow("a", "book-1", 1, 0, "Глава 1", "альфа бета гамма дельта"),
        ChunkRow("b", "book-1", 1, 1, "Глава 1", "альфа бета гамма дельта epsilon"),
    ]
    cases = [EvalCase("вопрос", ("a",), ("book-1",))]

    (annotation,) = build_multigold(cases, chunks)

    assert annotation.grades["b"] == GRADE_ADMISSIBLE
    assert annotation.reasons["b"] == "same_page"


def test_pages_collapse_with_max_grade() -> None:
    chunks = [
        ChunkRow("a", "book-1", 1, 0, "Глава 1", "текст"),
        ChunkRow("b", "book-1", 2, 1, "Глава 1", "текст"),
        ChunkRow("c", "book-1", 3, 5, "Глава 1", "текст"),
    ]
    cases = [EvalCase("вопрос", ("a",), ("book-1",))]

    (annotation,) = build_multigold(cases, chunks)
    pages = annotation.pages(
        {"a": ("book-1", 1), "b": ("book-1", 2), "c": ("book-1", 3)}
    )

    assert pages[("book-1", 1)] == GRADE_GOLD
    assert pages[("book-1", 2)] == GRADE_ADMISSIBLE
    assert ("book-1", 3) not in pages


def test_shingles_ignore_case_and_word_order_noise() -> None:
    left = shingles("Один два три четыре пять")
    right = shingles("один  два три четыре пять!")

    assert jaccard(left, right) == 1.0


def test_shingles_detect_partial_overlap() -> None:
    left = shingles("альфа бета гамма дельта")
    right = shingles("альфа бета гамма омега")

    assert 0.0 < jaccard(left, right) < 1.0
    assert jaccard(shingles("совсем другой текст"), left) == 0.0
