"""Тесты корреляции feedback ↔ runtime-качество (T104)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "feedback_correlation.py"
spec = importlib.util.spec_from_file_location("feedback_correlation", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pearson_detects_positive_and_negative_links() -> None:
    assert module._pearson([1.0, 1.0, 0.0, 0.0], [0.0, 0.1, 0.9, 1.0]) < -0.9
    assert module._pearson([1.0, 1.0, 0.0, 0.0], [0.9, 1.0, 0.0, 0.1]) > 0.9


def test_by_request_id_keeps_last_and_skips_empty() -> None:
    rows = [
        {"request_id": "a", "rating": "up"},
        {"request_id": "a", "rating": "down"},
        {"request_id": "", "rating": "up"},
    ]
    assert module._by_request_id(rows) == {"a": {"request_id": "a", "rating": "down"}}
