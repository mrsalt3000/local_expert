"""JSONL-трейсы агента без текста книги."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from book_expert.config import Settings


class TraceRecorder:
    """Append-only JSONL; по умолчанию пишет только метаданные."""

    REDACTED_KEYS = {"query", "queries", "plan", "text", "answer", "quote"}

    def __init__(self, path: Path, *, include_text: bool = False) -> None:
        self.path = path
        self.include_text = include_text
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, settings: Settings, prefix: str = "agent") -> TraceRecorder:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        return cls(
            settings.artifacts_dir / "traces" / f"{prefix}-{stamp}-{suffix}.jsonl",
            include_text=settings.trace_text,
        )

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._redact(fields),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _redact(self, fields: dict[str, Any]) -> dict[str, Any]:
        if self.include_text:
            return fields
        result: dict[str, Any] = {}
        for key, value in fields.items():
            if key in self.REDACTED_KEYS:
                result[key] = f"<{len(str(value))} chars>"
            else:
                result[key] = value
        return result


__all__ = ["TraceRecorder"]
