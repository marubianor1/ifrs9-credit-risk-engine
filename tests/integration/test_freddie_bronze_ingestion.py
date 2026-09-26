from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from ifrs9.ingestion.freddie import (
    FreddieIngestionError,
    read_member_to_frame,
    run_freddie_ingestion,
    validate_parquet_with_duckdb,
)
from ifrs9.ingestion.schema_registry import load_schema


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config" / "schemas").mkdir(parents=True)
    source_schema_dir = Path.cwd() / "config" / "schemas"
    for schema_path in source_schema_dir.glob("*.yaml"):
        shutil.copy(schema_path, repo / "config" / "schemas" / schema_path.name)
    (repo / "config" / "data.yaml").write_text(
        "\n".join(
            [
                "paths:",
                "  bronze_dir: data/bronze",
                "source_data:",
                "  freddie_mac:",
                "    source_dir: data/download",
                "ingestion:",
                "  fail_on_column_mismatch: true",
                "  bronze_freddie_dir: data/bronze/freddie",
                "  manifest_dir: artifacts/ingestion",
                "",
            ]
        )
    )
    (repo / "data" / "download").mkdir(parents=True)
    return repo


def _orig_row(loan_id: str = "F26Q10000001", vantagescore: str = "9999") -> str:
    values = [
        "799",
        "202603",
        "N",
        "205602",
        "19340",
        "0",
        "1",
        "P",
        "79",
        "43",
        "115000",
        "79",
        "6.125",
        "R",
        "N",
        "FRM",
        "IA",
        "CO",
        "527",
        loan_id,
        "N",
        "360",
        "1",
        "OTHER",
        "N",
        "",
        "",
        "N",
        "1",
        "N",
        vantagescore,
    ]
    return "|".join(values)


def _perf_row(loan_id: str = "F26Q10000001", period: str = "202602") -> str:
    values = [
        loan_id,
        period,
        "114000.00",
        "XX",
        "0",
        "360",
        "",
        "",
        "",
        "",
        "6.125",
        "0.00",
        "",
        "",
        "U",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "0.00",
        "",
        "",
        "999",
        "",
        "",
        "",
        "",
        "",
        "114000.00",
        "7",
        "ROCKET MORTGAGE, LLC",
        "0.00",
    ]
    return "|".join(values)


def _write_zip(repo: Path, year: int, orig_rows: list[str], perf_rows: list[str]) -> Path:
    zip_path = repo / "data" / "download" / f"sample_{year}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"sample_orig_{year}.txt", "\n".join(orig_rows) + "\n")
        archive.writestr(f"sample_perf_{year}.txt", "\n".join(perf_rows) + "\n")
    return zip_path


def test_positional_mapping_and_sentinel_preservation(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    zip_path = _write_zip(repo, 2026, [_orig_row()], [_perf_row()])
    schema = load_schema("origination", 2026, repo_root=repo)

    frame = read_member_to_frame(
        zip_path,
        "sample_orig_2026.txt",
        schema,
        year=2026,
        ingested_at="2026-09-26T00:00:00+00:00",
    )

    assert frame["loan_id"][0] == "F26Q10000001"
    assert frame["vantagescore_4"][0] == 9999
    assert frame["_source_row_number"][0] == 1
    assert frame["_source_zip"][0] == "sample_2026.zip"


def test_malformed_row_wrong_column_count_fails(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    zip_path = _write_zip(repo, 2026, ["a|b|c"], [_perf_row()])
    schema = load_schema("origination", 2026, repo_root=repo)

    with pytest.raises(FreddieIngestionError, match="expected 31"):
        read_member_to_frame(
            zip_path,
            "sample_orig_2026.txt",
            schema,
            year=2026,
            ingested_at="2026-09-26T00:00:00+00:00",
        )


def test_valid_ingestion_manifest_parquet_duckdb_and_dq(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_zip(
        repo,
        2026,
        [_orig_row(), _orig_row()],
        [_perf_row(period="202602"), _perf_row(period="202602"), _perf_row("UNKNOWN", "202603")],
    )

    results = run_freddie_ingestion(year=2026, force=False, repo_root=repo)

    assert results[0].status == "success"
    manifest = json.loads((repo / "artifacts" / "ingestion" / "bronze_manifest.json").read_text())
    assert manifest["results"][0]["origination_rows"] == 2
    assert manifest["results"][0]["performance_rows"] == 3
    dq_report = (repo / "artifacts" / "ingestion" / "bronze_data_quality.csv").read_text()
    assert "duplicate_loan_ids" in dq_report
    assert "performance_loans_without_origination" in dq_report

    parquet_path = (
        repo
        / "data"
        / "bronze"
        / "freddie"
        / "performance"
        / "vintage_year=2026"
        / "part-performance.parquet"
    )
    duck = validate_parquet_with_duckdb(parquet_path, "performance")
    assert duck["row_count"] == 3
    assert duck["min_period"] == "202602"
    assert duck["max_period"] == "202603"


def test_idempotent_rerun_and_force_behavior(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_zip(repo, 2026, [_orig_row()], [_perf_row()])
    run_freddie_ingestion(year=2026, force=False, repo_root=repo)

    skipped = run_freddie_ingestion(year=2026, force=False, repo_root=repo)
    forced = run_freddie_ingestion(year=2026, force=True, repo_root=repo)

    assert skipped[0].status == "skipped_existing"
    assert forced[0].status == "success"
