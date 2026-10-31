"""Audit tool-calls: append-only JSONL без содержимого файлов и книг."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from book_expert.config import Settings

REDACTED_KEYS = {"content", "text", "query", "question"}
ARG_LIMIT = 300


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, settings: Settings) -> AuditLogger:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return cls(settings.audit_dir / f"tools-{stamp}.jsonl")

    def log(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        ok: bool,
        exit_code: int | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "args": _redact(args),
            "ok": ok,
            "exit_code": exit_code,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "error": error[:ARG_LIMIT] if error else None,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _redact(args: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if key in REDACTED_KEYS:
            text = str(value)
            redacted[key] = f"<{len(text)} chars>"
        elif isinstance(value, str):
            redacted[key] = value[:ARG_LIMIT]
        else:
            redacted[key] = value
    return redacted


__all__ = ["AuditLogger"]
