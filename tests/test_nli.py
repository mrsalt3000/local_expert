"""Тесты NLI-обвязки без загрузки модели."""

from __future__ import annotations

from book_expert.eval.nli import NLICitationScorer


class FakeScorer(NLICitationScorer):
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score_many(self, pairs):  # type: ignore[override]
        pairs = list(pairs)
        return self._scores[: len(pairs)]


def test_nli_supports_uses_max_across_cited_excerpts() -> None:
    scorer = FakeScorer([0.2, 0.8])

    assert scorer.supports("claim", ["a", "b"], threshold=0.5) is True
    assert scorer.supports("claim", ["a", "b"], threshold=0.9) is False


def test_nli_supports_false_without_excerpts() -> None:
    assert FakeScorer([]).supports("claim", [], threshold=0.0) is False
