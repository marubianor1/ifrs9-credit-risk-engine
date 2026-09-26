from __future__ import annotations

from pathlib import Path

import pytest

from ifrs9.ingestion.schema_registry import SchemaRegistryError, load_schema


def test_load_origination_schema_for_supported_year() -> None:
    schema = load_schema("origination", year=2015)

    assert schema.schema_id == "freddie_origination_v1"
    assert schema.expected_column_count == 31
    assert schema.columns[0].canonical_name == "credit_score"


def test_load_performance_schema_for_supported_year() -> None:
    schema = load_schema("performance", year=2026)

    assert schema.schema_id == "freddie_performance_v1"
    assert schema.expected_column_count == 35
    assert schema.columns[0].canonical_name == "loan_id"
    assert schema.columns[32].canonical_name == "mortgage_insurance_cancellation_indicator"
    assert schema.columns[33].canonical_name == "servicer_name"
    assert schema.columns[34].canonical_name == "bankruptcy_cramdown_costs"


def test_load_schema_rejects_unsupported_year() -> None:
    with pytest.raises(SchemaRegistryError, match="No Freddie Mac origination schema"):
        load_schema("origination", year=1998)


def test_load_schema_rejects_missing_registry(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")

    with pytest.raises(SchemaRegistryError, match="Schema version registry not found"):
        load_schema("performance", year=2026, repo_root=tmp_path)
