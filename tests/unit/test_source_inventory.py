from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ifrs9.ingestion.source_inventory import (
    SourceInventoryError,
    identify_sample_members,
    inspect_sample_zip,
    parse_sample_year,
    split_pipe_row,
    validate_column_count,
)


def test_split_pipe_row_preserves_empty_fields() -> None:
    assert split_pipe_row("a||c|\n") == ["a", "", "c", ""]


def test_validate_column_count_detects_incorrect_width() -> None:
    with pytest.raises(SourceInventoryError, match="Expected 4 columns"):
        validate_column_count("a|b|c", expected_column_count=4)


def test_parse_sample_year_from_valid_filename() -> None:
    assert parse_sample_year("sample_2026.zip") == 2026


def test_parse_sample_year_rejects_malformed_filename() -> None:
    with pytest.raises(SourceInventoryError, match="Unsupported"):
        parse_sample_year("historical_data_2026Q1.zip")


def test_identify_sample_members_for_year() -> None:
    members = ("sample_orig_2026.txt", "sample_perf_2026.txt")
    assert identify_sample_members(members, 2026) == members


def test_inspect_sample_zip_detects_member_widths(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample_2026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample_orig_2026.txt", "a|b|c\n1|2|3\n")
        archive.writestr("sample_perf_2026.txt", "a|b\n1|2\n")

    profile = inspect_sample_zip(zip_path, sample_size=10)

    assert profile.year == 2026
    assert profile.origination.sampled_column_count == 3
    assert profile.performance.sampled_column_count == 2
    assert profile.origination.consistent is True
    assert profile.performance.consistent is True


def test_inspect_sample_zip_detects_inconsistent_widths(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample_2026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample_orig_2026.txt", "a|b|c\n1|2\n")
        archive.writestr("sample_perf_2026.txt", "a|b\n1|2\n")

    profile = inspect_sample_zip(zip_path, sample_size=10)

    assert profile.origination.consistent is False
    assert profile.origination.observed_column_counts == (2, 3)
