"""Contract tests for the parser assessment schema and synthetic fixture.

Только стандартная библиотека: без jsonschema, pydantic и других зависимостей.
Local-only manifest data/eval/parser_assessment.local.json не читается.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "parser-assessment.schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "parser_assessment.synthetic.json"

MANIFEST_SCHEMA_PATH = ROOT / "schemas" / "parser-assessment-manifest.schema.json"
MANIFEST_FIXTURE_PATH = (
    ROOT / "tests" / "fixtures" / "parser_assessment_manifest.synthetic.json"
)

EXPECTED_MANIFEST_REQUIRED = {
    "manifest_schema_version",
    "assessment_schema_version",
    "document_ref",
    "records",
}

EXPECTED_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"

EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "parser",
    "document_ref",
    "page_ref",
    "page_types",
    "criteria",
    "review",
    "runtime",
}

EXPECTED_CRITERIA = {
    "text_coverage",
    "reading_order",
    "heading_structure",
    "table_fidelity",
    "footnote_separation",
    "header_footer_handling",
    "image_handling",
    "page_linkage",
    "runtime",
}

EXPECTED_STATUSES = {"pass", "warning", "fail", "not_applicable"}

EXPECTED_PAGE_TYPES = {
    "body_text",
    "mixed_language",
    "table_of_contents",
    "table",
    "footnote",
    "header_footer",
    "image_or_diagram",
    "callout",
    "heading_hierarchy",
    "end_boundary",
}

BANNED_FIELDS = {
    "source_text",
    "extracted_text",
    "ocr_text",
    "quote",
    "passage",
    "chunk_text",
    "pdf_path",
    "absolute_path",
}


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _property_names(node: object) -> set[str]:
    """Все имена из всех "properties" в schema (рекурсивно)."""
    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for value in node.values():
            names.update(_property_names(value))
    elif isinstance(node, list):
        for item in node:
            names.update(_property_names(item))
    return names


def _all_keys(node: object) -> set[str]:
    """Все ключи JSON-документа (рекурсивно)."""
    keys: set[str] = set()
    if isinstance(node, dict):
        keys.update(node)
        for value in node.values():
            keys.update(_all_keys(value))
    elif isinstance(node, list):
        for item in node:
            keys.update(_all_keys(item))
    return keys


schema = _load_json(SCHEMA_PATH)
fixture = _load_json(FIXTURE_PATH)

manifest_schema = _load_json(MANIFEST_SCHEMA_PATH)
manifest_fixture = _load_json(MANIFEST_FIXTURE_PATH)


def test_schema_uri() -> None:
    assert schema["$schema"] == EXPECTED_SCHEMA_URI


def test_top_level_required_fields() -> None:
    assert set(schema["required"]) == EXPECTED_TOP_LEVEL_FIELDS


def test_top_level_additional_properties_is_false() -> None:
    assert schema["additionalProperties"] is False


def test_status_enum_is_exact() -> None:
    assert set(schema["$defs"]["status"]["enum"]) == EXPECTED_STATUSES


def test_criteria_contains_all_nine_names() -> None:
    criteria = schema["properties"]["criteria"]
    assert set(criteria["required"]) == EXPECTED_CRITERIA
    assert set(criteria["properties"]) == EXPECTED_CRITERIA


def test_each_criterion_references_status_enum() -> None:
    criteria_properties = schema["properties"]["criteria"]["properties"]
    for name in EXPECTED_CRITERIA:
        assert criteria_properties[name] == {"$ref": "#/$defs/status"}


def test_page_types_enum_is_exact() -> None:
    enum = schema["properties"]["page_types"]["items"]["enum"]
    assert set(enum) == EXPECTED_PAGE_TYPES


def test_banned_fields_absent_from_schema_properties() -> None:
    assert _property_names(schema).isdisjoint(BANNED_FIELDS)


def test_banned_fields_absent_from_fixture() -> None:
    assert _all_keys(fixture).isdisjoint(BANNED_FIELDS)


def test_fixture_shape_matches_schema() -> None:
    assert set(fixture) == set(schema["properties"])


def test_fixture_criteria_statuses_are_valid() -> None:
    for status in fixture["criteria"].values():
        assert status in EXPECTED_STATUSES


def test_fixture_page_types_are_valid_and_non_empty() -> None:
    assert fixture["page_types"]
    assert set(fixture["page_types"]) <= EXPECTED_PAGE_TYPES


def test_fixture_elapsed_ms_is_non_negative_integer() -> None:
    elapsed_ms = fixture["runtime"]["elapsed_ms"]
    assert isinstance(elapsed_ms, int) and elapsed_ms >= 0


def test_manifest_schema_uri() -> None:
    assert manifest_schema["$schema"] == EXPECTED_SCHEMA_URI


def test_manifest_schema_has_no_id() -> None:
    assert "$id" not in manifest_schema


def test_manifest_required_fields_are_exact() -> None:
    assert set(manifest_schema["required"]) == EXPECTED_MANIFEST_REQUIRED


def test_manifest_versions_are_const() -> None:
    assert manifest_schema["properties"]["manifest_schema_version"] == {
        "type": "string",
        "const": "1.0.0",
    }
    assert manifest_schema["properties"]["assessment_schema_version"] == {
        "type": "string",
        "const": "1.0.0",
    }


def test_manifest_note_is_optional_string() -> None:
    assert "manifest_note" in manifest_schema["properties"]
    assert "manifest_note" not in EXPECTED_MANIFEST_REQUIRED
    assert manifest_schema["properties"]["manifest_note"] == {"type": "string"}


def test_manifest_additional_properties_is_false() -> None:
    assert manifest_schema["additionalProperties"] is False


def test_manifest_records_items_refers_to_page_level_schema() -> None:
    items = manifest_schema["properties"]["records"]["items"]
    assert items == {"$ref": "./parser-assessment.schema.json"}


def test_manifest_banned_fields_absent_from_schema_properties() -> None:
    assert _property_names(manifest_schema).isdisjoint(BANNED_FIELDS)


def test_manifest_fixture_top_level_keys_match_schema_properties() -> None:
    assert set(manifest_fixture) == set(manifest_schema["properties"])


def test_manifest_fixture_records_non_empty() -> None:
    assert manifest_fixture["records"]


def test_manifest_fixture_first_record_has_page_level_keys() -> None:
    first_record = manifest_fixture["records"][0]
    assert set(first_record) == set(schema["properties"])


def test_manifest_fixture_banned_fields_absent() -> None:
    assert _all_keys(manifest_fixture).isdisjoint(BANNED_FIELDS)
