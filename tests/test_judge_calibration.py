"""Тесты метрик калибровки судьи T86."""

from __future__ import annotations

from book_expert.eval.calibration import binary_metrics, sample_claims


def test_binary_metrics_perfect_agreement() -> None:
    result = binary_metrics([(True, True), (False, False)])

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["kappa"] == 1.0


def test_binary_metrics_all_positive_predictions() -> None:
    result = binary_metrics([(True, True), (False, True)])

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["kappa"] == 0.0


def test_binary_metrics_empty() -> None:
    assert binary_metrics([]) == {"n": 0}


def test_sample_claims_is_deterministic_and_prefers_cited() -> None:
    claims = [
        {
            "id": f"1:{index}",
            "case": 1,
            "cited": [1] if index <= 6 else [],
        }
        for index in range(1, 11)
    ]

    first = sample_claims(claims, 4)
    second = sample_claims(claims, 4)

    assert [claim["id"] for claim in first] == [claim["id"] for claim in second]
    assert len(first) == 4
    assert sum(1 for claim in first if claim["cited"]) >= 2
