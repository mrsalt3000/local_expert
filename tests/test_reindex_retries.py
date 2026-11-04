"""Повтор upsert при таймауте Qdrant (T82)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reindex_embeddings.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reindex_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlakyClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("timed out")


def test_upsert_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    client = FlakyClient(failures=2)

    module._upsert_with_retries(client, "books", [], attempts=4)

    assert client.calls == 3


def test_upsert_reraises_after_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    client = FlakyClient(failures=10)

    with pytest.raises(TimeoutError):
        module._upsert_with_retries(client, "books", [], attempts=3)

    assert client.calls == 3
