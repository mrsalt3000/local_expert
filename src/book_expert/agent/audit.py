"""Audit tool-calls: append-only JSONL с hash chain и redaction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from book_expert.config import Settings

REDACTED_KEYS = {"content", "text", "query", "question"}
ARG_LIMIT = 300
GENESIS_HASH = "0" * 64


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash, self._seq = _tail_state(path)

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
            "seq": self._seq,
            "prev_hash": self._prev_hash,
            "tool": tool,
            "args": _redact(args),
            "ok": ok,
            "exit_code": exit_code,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "error": error[:ARG_LIMIT] if error else None,
        }
        digest = _digest(record)
        record["hash"] = digest
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._prev_hash = digest
        self._seq += 1


def verify_audit_chain(path: Path) -> tuple[bool, int]:
    """Проверяет hash chain; legacy-записи до неё пропускаются."""
    prev_hash = GENESIS_HASH
    expected_seq = 1
    count = 0
    started = False
    saw_content = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            saw_content = True
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False, count
            if "hash" not in record:
                if started:
                    return False, count
                continue
            if not started:
                started = True
            digest = record.pop("hash", None)
            if record.get("prev_hash") != prev_hash:
                return False, count
            if record.get("seq") != expected_seq:
                return False, count
            if not isinstance(digest, str) or digest != _digest(record):
                return False, count
            prev_hash = digest
            expected_seq += 1
            count += 1
    if saw_content and not started:
        return False, count
    return True, count


def _tail_state(path: Path) -> tuple[str, int]:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH, 1
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        digest = record.get("hash")
        seq = record.get("seq")
        if isinstance(digest, str) and isinstance(seq, int):
            return digest, seq + 1
    return GENESIS_HASH, 1


def _digest(record: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


__all__ = ["AuditLogger", "verify_audit_chain"]
