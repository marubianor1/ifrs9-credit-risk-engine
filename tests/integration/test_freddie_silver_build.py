from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from ifrs9.ingestion.freddie import run_freddie_ingestion
from ifrs9.transformations.freddie_silver import run_freddie_silver_build


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
                "source_data:",
                "  freddie_mac:",
                "    source_dir: data/download",
                "ingestion:",
                "  bronze_freddie_dir: data/bronze/freddie",
                "  silver_freddie_dir: data/silver/freddie",
                "  manifest_dir: artifacts/ingestion",
                "  silver_manifest_dir: artifacts/silver",
                "",
            ]
        )
    )
    (repo / "data" / "download").mkdir(parents=True)
    return repo


def _orig_row(loan_id: str = "F26Q10000001") -> str:
    return "|".join(
        [
            "799",
            "202603",
            "N",
            "205602",
            "19340",
            "0",
            "1",
            "P",
            "79",
            "999",
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
            "9999",
        ]
    )


def _perf_row(loan_id: str = "F26Q10000001", period: str = "202602") -> str:
    values = [
        loan_id,
        period,
        "114000.00",
        "03",
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
        "OTHER",
        "0.00",
    ]
    return "|".join(values)


def _write_zip(repo: Path) -> None:
    with zipfile.ZipFile(repo / "data" / "download" / "sample_2026.zip", "w") as archive:
        archive.writestr("sample_orig_2026.txt", _orig_row() + "\n")
        archive.writestr(
            "sample_perf_2026.txt",
            _perf_row(period="202602") + "\n" + _perf_row(period="202604") + "\n",
        )


def test_silver_build_preserves_rows_lineage_manifest_and_force(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_zip(repo)
    run_freddie_ingestion(year=2026, force=True, repo_root=repo)

    first = run_freddie_silver_build(year=2026, force=False, repo_root=repo)
    skipped = run_freddie_silver_build(year=2026, force=False, repo_root=repo)
    forced = run_freddie_silver_build(year=2026, force=True, repo_root=repo)

    assert {result.status for result in first} == {"success"}
    assert {result.status for result in skipped} == {"skipped_existing"}
    assert {result.status for result in forced} == {"success"}

    manifest = json.loads((repo / "artifacts" / "silver" / "silver_manifest.json").read_text())
    assert manifest["results"][0]["bronze_input_rows"] == 1
    assert manifest["results"][0]["silver_output_rows"] == 1

    summary = (repo / "artifacts" / "silver" / "normalization_summary.csv").read_text()
    assert "original_debt_to_income_ratio" in summary
    assert "estimated_loan_to_value" in summary

    temporal = (repo / "artifacts" / "silver" / "performance_temporal_gaps.csv").read_text()
    assert "loans_with_month_gaps,1" in temporal
