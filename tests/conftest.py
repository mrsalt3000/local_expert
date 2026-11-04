"""Общие фикстуры тестов: изоляция от экспортированных BOOK_EXPERT_*."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_book_expert_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Экспортированные переменные shell не должны влиять на тесты."""
    for key in list(os.environ):
        if key.startswith("BOOK_EXPERT_"):
            monkeypatch.delenv(key, raising=False)
