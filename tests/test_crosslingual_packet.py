"""Тесты сэмплера и оценки cross-lingual NLI-пакета (T100)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packet_builder = _load(
    "build_crosslingual_packet", ROOT / "scripts" / "build_crosslingual_packet.py"
)
eval_nli = _load("eval_nli", ROOT / "scripts" / "eval_nli.py")


def _args(**overrides: Any) -> SimpleNamespace:
    values = {"min_length": 50, "max_length": 500}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ascii_share_rejects_mojibake() -> None:
    assert packet_builder._ascii_share("Plain English text") == 1.0
    assert packet_builder._ascii_share("Áåäíûé Ïàïà Ðîáåðò") < 0.1


def test_eligible_requires_english_prose_and_body_page() -> None:
    text = "The system is reliable and can handle the load of many users."
    assert packet_builder._eligible(text, 50, 100, _args()) is True
    assert packet_builder._eligible(text, 5, 100, _args()) is False
    assert (
        packet_builder._eligible("Table of Contents " + text, 50, 100, _args()) is False
    )
    assert packet_builder._eligible("x" * 500, 50, 100, _args()) is False


def test_build_packet_writes_blind_labels_and_gold(tmp_path: Path) -> None:
    (tmp_path / "candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "chunk_id": "chk_a",
                        "book_id": "book",
                        "book_title": "Book",
                        "page": 10,
                        "section": None,
                        "text": "This English excerpt is long enough to be detected as English prose.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    claims = [
        {
            "chunk_id": "chk_a",
            "claim": "Первое русское утверждение достаточно длинное для определения языка.",
            "label": True,
        },
        {
            "chunk_id": "chk_a",
            "claim": "Второе русское утверждение тоже достаточно длинное для проверки.",
            "label": False,
        },
    ]
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(claims, ensure_ascii=False), encoding="utf-8")

    packet_builder._build_packet(tmp_path, claims_path, seed=1)

    packet = json.loads((tmp_path / "packet.json").read_text(encoding="utf-8"))
    assert [claim["id"] for claim in packet["claims"]] == ["x01", "x02"]
    assert all(
        claim["excerpts"][0]["text"]
        == "This English excerpt is long enough to be detected as English prose."
        for claim in packet["claims"]
    )
    gold = json.loads(
        (tmp_path / "private" / "gold.local.json").read_text(encoding="utf-8")
    )
    assert sorted(gold["labels"].values()) == [False, True]
    labels = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig")
    assert labels.splitlines()[0].startswith("id;label;claim;")
    rows = labels.splitlines()[1:]
    assert all(row.split(";")[1] == "" for row in rows)


def test_cross_lingual_metrics_with_fake_scorer(tmp_path: Path) -> None:
    packet = {
        "claims": [
            {"id": "x01", "text": "A", "excerpts": [{"marker": 1, "text": "a"}]},
            {"id": "x02", "text": "B", "excerpts": [{"marker": 1, "text": "b"}]},
        ]
    }
    (tmp_path / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    (tmp_path / "labels.csv").write_text(
        "id;label\nx01;true\nx02;false\n", encoding="utf-8-sig"
    )

    class FakeScorer:
        def score_many(self, pairs: Any) -> list[float]:
            return [0.9 if hypothesis == "A" else 0.1 for _, hypothesis in pairs]

    result = eval_nli._cross_lingual(
        FakeScorer(), tmp_path / "packet.json", tmp_path / "labels.csv", 0.5
    )
    assert result["claims"] == 2
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["kappa"] == 1.0
