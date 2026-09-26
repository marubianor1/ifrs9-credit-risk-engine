from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from ifrs9.development.config import load_development_config
from ifrs9.development.factory import run_development_sample_build


def _repo_fixture(tmp_path: Path, *, sampling_strategy: str = "none") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config" / "features").mkdir(parents=True)
    shutil.copy(
        Path.cwd() / "config" / "features" / "point_in_time_features.yaml",
        repo / "config" / "features" / "point_in_time_features.yaml",
    )
    (repo / "config" / "data.yaml").write_text(
        "\n".join(
            [
                "ingestion:",
                "  gold_freddie_dir: data/gold/freddie",
                "",
            ]
        )
    )
    development_config = f"""
development_sample:
  version: test
  reporting_cutoff: 2026-03-01
  binary_12m_full_horizon_cutoff: 2025-03-01
  model_use: pd
  population: behavioural
  snapshot_frequency: quarter_end
  split_windows:
    train:
      start: 2012-01-01
      end: 2018-12-01
    validation:
      start: 2019-01-01
      end: 2021-12-01
    oot:
      start: 2022-01-01
      end: 2025-03-01
  sampling:
    strategy: {sampling_strategy}
    seed: 7
    nondefault_sample_fraction: 0.5
    minimum_nondefaults_per_stratum: 0
  loan_disjoint: false
  notes: []
"""
    (repo / "config" / "development_samples.yaml").write_text(development_config)
    return repo


def _write_targets(repo: Path) -> None:
    target_dir = repo / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
    target_dir.mkdir(parents=True)
    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM (
                VALUES
                ('L1', DATE '2018-03-01', true, true, 0, false, 'CENSORED'),
                ('L1', DATE '2018-04-01', true, true, 0, false, 'CENSORED'),
                ('L1', DATE '2019-03-01', true, true, 1, false, 'DEFAULT'),
                ('L1', DATE '2022-03-01', true, true, 0, false, 'CENSORED'),
                ('L2', DATE '2018-12-01', true, true, 1, false, 'DEFAULT'),
                ('L2', DATE '2021-12-01', true, true, 0, false, 'CENSORED'),
                ('L3', DATE '2025-03-01', true, true, 0, false, 'CENSORED'),
                ('L3', DATE '2025-06-01', true, true, 1, false, 'DEFAULT'),
                ('L4', DATE '2024-12-01', true, false, NULL, true, 'CENSORED'),
                ('L5', DATE '2020-12-01', false, true, NULL, false, 'CENSORED')
              ) AS t(
                loan_id, as_of_date, eligible_for_12m_pd, target_12m_observable,
                default_next_12m, event_censored, event_type
              )
            )
            TO '{(target_dir / "part-pd_12m_targets.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )


def _read_development(repo: Path) -> list[tuple]:
    output = repo / "data" / "gold" / "freddie" / "development"
    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            f"""
            SELECT loan_id, as_of_date, split, development_eligible_12m,
                   default_next_12m, sample_selected, sampling_probability,
                   sample_weight
            FROM read_parquet('{output.as_posix()}/*.parquet')
            ORDER BY loan_id, as_of_date
            """
        ).fetchall()


def test_development_sample_cutoff_splits_snapshots_and_feature_manifest(
    tmp_path: Path,
) -> None:
    repo = _repo_fixture(tmp_path)
    _write_targets(repo)

    result = run_development_sample_build(force=True, repo_root=repo)

    assert result.rows == 6
    rows = _read_development(repo)
    assert ("L1", rows[0][1], "TRAIN", True, 0, True, 1.0, 1.0) in rows
    assert all(row[1].month in {3, 6, 9, 12} for row in rows)
    assert any(row[0] == "L3" and str(row[1]) == "2025-03-01" and row[3] for row in rows)
    assert not any(str(row[1]) > "2025-03-01" and row[3] for row in rows)
    assert any(row[0] == "L4" and not row[3] for row in rows)

    manifest_path = repo / "artifacts" / "development" / "development_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["validation"]["max_eligible_as_of_date"] == "2025-03-01"
    assert manifest["validation"]["loans_in_multiple_splits"] == 2
    assert "actual_loss" not in manifest["approved_features"]
    assert "_gold_processed_at" not in manifest["approved_features"]
    assert (repo / "artifacts" / "development" / "split_profile.csv").exists()
    assert (repo / "artifacts" / "development" / "sample_profile.csv").exists()


def test_monthly_application_and_loan_disjoint_modes(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_targets(repo)

    monthly = run_development_sample_build(
        force=True,
        repo_root=repo,
        snapshot_frequency="monthly",
    )
    application = run_development_sample_build(
        force=True,
        repo_root=repo,
        population="application",
    )
    loan_disjoint = run_development_sample_build(
        force=True,
        repo_root=repo,
        loan_disjoint=True,
    )

    assert monthly.rows > application.rows
    assert application.rows == 3
    manifest_path = repo / "artifacts" / "development" / "development_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert loan_disjoint.rows == manifest["validation"]["eligible_rows"]
    assert manifest["validation"]["loans_in_multiple_splits"] == 0


def test_deterministic_nondefault_sampling_weights_and_positive_retention(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path, sampling_strategy="random_nondefault")
    _write_targets(repo)

    run_development_sample_build(force=True, repo_root=repo)
    first = _read_development(repo)
    run_development_sample_build(force=True, repo_root=repo)
    second = _read_development(repo)

    assert first == second
    assert all(row[5] for row in first if row[3] and row[4] == 1)
    assert all(row[6] == 1.0 and row[7] == 1.0 for row in first if row[3] and row[4] == 1)
    assert any(row[6] == 0.5 and row[7] == 2.0 for row in first if row[3] and row[4] == 0)


def test_config_rejects_overlapping_split_windows(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    config_path = repo / "config" / "development_samples.yaml"
    config_path.write_text(
        config_path.read_text().replace("start: 2019-01-01", "start: 2018-06-01")
    )

    try:
        load_development_config(repo)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Expected overlapping split windows to fail validation")
