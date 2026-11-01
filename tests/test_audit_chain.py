"""Тесты hash chain для audit-лога."""

from __future__ import annotations

import json
from pathlib import Path

from book_expert.agent.audit import AuditLogger, GENESIS_HASH, verify_audit_chain


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audit_chain_links_records_and_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = AuditLogger(path)
    first.log("rag_search", {"query": "secret"}, ok=True)

    second = AuditLogger(path)
    second.log("rag_search", {"query": "another"}, ok=True)

    records = _records(path)
    assert [record["seq"] for record in records] == [1, 2]
    assert records[0]["prev_hash"] == GENESIS_HASH
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert all(len(record["hash"]) == 64 for record in records)
    assert records[0]["args"]["query"] == "<6 chars>"
    assert verify_audit_chain(path) == (True, 2)


def test_audit_chain_skips_legacy_records_before_first_hash(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"tool": "legacy", "ok": true}\n', encoding="utf-8")

    logger = AuditLogger(path)
    logger.log("rag_search", {"query": "new"}, ok=True)

    assert verify_audit_chain(path) == (True, 1)


def test_audit_chain_reports_legacy_only_as_unverified(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"tool": "legacy", "ok": true}\n', encoding="utf-8")

    assert verify_audit_chain(path) == (False, 0)


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    logger.log("rag_search", {"query": "one"}, ok=True)
    logger.log("rag_search", {"query": "two"}, ok=True)

    records = _records(path)
    records[1]["tool"] = "tampered"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    assert verify_audit_chain(path) == (False, 1)
