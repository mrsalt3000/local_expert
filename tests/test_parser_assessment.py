"""Тесты local-only parser assessment на synthetic PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from book_expert.eval.parser_assessment import (
    AssessmentError,
    build_manifest,
    parse_page_range,
    write_manifest,
)


def _pdf(path: Path) -> Path:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Chapter 1", fontsize=12)
    first.insert_text((72, 96), "Body text about retrieval.", fontsize=11)
    first.insert_text((72, first.rect.height - 40), "7", fontsize=10)
    document.new_page()
    contents = document.new_page()
    y = 72.0
    for line in (
        "Chapter 1 ..... 1",
        "Chapter 2 ..... 5",
        "Chapter 3 ..... 9",
        "Chapter 4 ..... 12",
        "Chapter 5 ..... 15",
    ):
        contents.insert_text((72, y), line, fontsize=11)
        y += 22
    document.save(path)
    document.close()
    return path


def test_build_manifest_is_local_only(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "synthetic.pdf")
    output = tmp_path / "assessment.local.json"

    manifest = build_manifest(pdf, output=output)

    assert manifest["manifest_schema_version"] == "1.0.0"
    assert manifest["assessment_schema_version"] == "1.0.0"
    assert len(manifest["records"]) == 3
    assert output.exists()
    text = json.dumps(manifest, ensure_ascii=False)
    assert str(tmp_path) not in text
    assert "Body text about retrieval." not in text
    assert "extracted_text" not in text


def test_page_records_have_coverage_and_page_linkage(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "synthetic.pdf")
    manifest = build_manifest(pdf)
    first, empty, contents = manifest["records"]

    assert first["page_ref"]["pdf_page_index"] == 0
    assert first["page_ref"]["printed_page_label"] == "7"
    assert first["criteria"]["text_coverage"] == "pass"
    assert first["criteria"]["page_linkage"] == "pass"
    assert first["page_types"] == ["body_text"]

    assert empty["page_ref"]["pdf_page_index"] == 1
    assert empty["criteria"]["text_coverage"] == "fail"
    assert empty["page_types"] == ["image_or_diagram"]

    assert contents["page_types"] == ["table_of_contents"]

    for record in manifest["records"]:
        assert record["criteria"]["page_linkage"] == "pass"
        assert isinstance(record["runtime"]["elapsed_ms"], int)
        assert record["parser"]["id"] == "pymupdf"
        assert record["parser"]["configuration_hash"]


def test_page_selection(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "synthetic.pdf")
    manifest = build_manifest(pdf, pages=[2])
    assert len(manifest["records"]) == 1
    assert manifest["records"][0]["page_ref"]["pdf_page_index"] == 2


def test_write_manifest_refuses_tracked_path(tmp_path: Path) -> None:
    with pytest.raises(AssessmentError, match="local"):
        write_manifest({"records": []}, tmp_path / "assessment.json")


def test_parse_page_range() -> None:
    assert parse_page_range("1-3,5") == [0, 1, 2, 4]
    assert parse_page_range("") == []
    with pytest.raises(ValueError, match="invalid page range"):
        parse_page_range("0-2")
