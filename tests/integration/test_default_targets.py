from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from ifrs9.targets.factory import run_target_build


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config").mkdir()
    shutil.copy(Path.cwd() / "config" / "default_definition.yaml", repo / "config")
    default_config = repo / "config" / "default_definition.yaml"
    default_config.write_text(
        default_config.read_text().replace(
            "reporting_cutoff: 2026-03-01",
            "reporting_cutoff: 2027-03-01",
        )
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
    return repo


def _write_gold_fixture(repo: Path, *, future_defaults: bool = True) -> None:
    gold = repo / "data" / "gold" / "freddie"
    month = gold / "loan_month" / "vintage_year=2026"
    static = gold / "loan_static" / "vintage_year=2026"
    month.mkdir(parents=True)
    static.mkdir(parents=True)
    delinquency_for_l1_march = 3 if future_defaults else 0
    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM (
                VALUES
                ('L1', 2026, DATE '2026-01-01', 0, '0', NULL, NULL, false, 0.0),
                ('L1', 2026, DATE '2026-02-01', 0, '0', NULL, NULL, false, 0.0),
                ('L1', 2026, DATE '2026-03-01', {delinquency_for_l1_march},
                 '{delinquency_for_l1_march}', NULL, NULL, false, 0.0),
                ('L2', 2026, DATE '2026-01-01', 0, '0', NULL, NULL, false, 0.0),
                ('L2', 2026, DATE '2026-02-01', 0, '0', '01', DATE '2026-02-01',
                 false, 0.0),
                ('L3', 2026, DATE '2026-11-01', 0, '0', NULL, NULL, false, 0.0),
                ('L4', 2026, DATE '2026-01-01', 3, '3', NULL, NULL, false, 0.0),
                ('L4', 2026, DATE '2026-02-01', 0, '0', NULL, NULL, false, 0.0),
                ('L4', 2026, DATE '2026-03-01', 0, '0', NULL, NULL, false, 0.0),
                ('L4', 2026, DATE '2026-04-01', 0, '0', NULL, NULL, false, 0.0),
                ('L4', 2026, DATE '2026-06-01', 3, '3', NULL, NULL, true, 0.0),
                ('L5', 2026, DATE '2026-01-01', 0, '0', '09', DATE '2026-01-01',
                 false, 0.0),
                ('L6', 2026, DATE '2026-01-01', 0, '0', NULL, NULL, false, 9999.0)
              ) AS t(
                loan_id, vintage_year, as_of_date, delinquency_months,
                delinquency_status_raw, zero_balance_code, zero_balance_effective_date,
                has_reporting_gap_to_date, actual_loss
              )
            )
            TO '{(month / "part-loan_month.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM (
                VALUES
                ('L1', 2026, TIMESTAMP '2026-01-01 00:00:00'),
                ('L2', 2026, TIMESTAMP '2026-01-01 00:00:00'),
                ('L3', 2026, TIMESTAMP '2026-01-01 00:00:00'),
                ('L4', 2026, TIMESTAMP '2026-01-01 00:00:00'),
                ('L5', 2026, TIMESTAMP '2026-01-01 00:00:00'),
                ('L6', 2026, TIMESTAMP '2026-01-01 00:00:00')
              ) AS t(loan_id, vintage_year, _gold_processed_at)
            )
            TO '{(static / "part-loan_static.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )


def test_default_targets_handle_backdating_censoring_exits_and_cure(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_gold_fixture(repo)

    result = run_target_build(all_targets=True, force=True, repo_root=repo)

    assert result.pd_12m_rows == 13
    with duckdb.connect(database=":memory:") as con:
        target_path = repo / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
        event_path = repo / "data" / "gold" / "freddie" / "targets" / "default_events"
        target_glob = f"{target_path.as_posix()}/*.parquet"
        event_glob = f"{event_path.as_posix()}/*.parquet"
        con.execute(
            f"CREATE VIEW targets AS SELECT * FROM read_parquet('{target_glob}')"
        )
        con.execute(
            f"CREATE VIEW events AS SELECT * FROM read_parquet('{event_glob}')"
        )

        assert con.execute(
            """
            SELECT default_flag, default_next_12m, months_to_default_from_observation
            FROM targets
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-01-01'
            """
        ).fetchone() == (False, 1, 2)
        assert con.execute(
            """
            SELECT default_flag, eligible_for_12m_pd
            FROM targets
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone() == (True, False)
        assert con.execute(
            """
            SELECT default_flag, event_type, default_next_12m, target_12m_observable
            FROM targets
            WHERE loan_id = 'L2' AND as_of_date = DATE '2026-01-01'
            """
        ).fetchone() == (False, "Prepaid or Matured", 0, True)
        assert con.execute(
            """
            SELECT default_next_12m, target_12m_observable
            FROM targets
            WHERE loan_id = 'L3' AND as_of_date = DATE '2026-11-01'
            """
        ).fetchone() == (None, False)
        assert con.execute(
            """
            SELECT default_flag, default_reason
            FROM targets
            WHERE loan_id = 'L5'
            """
        ).fetchone() == (True, "REO_DISPOSITION")
        assert con.execute(
            """
            SELECT default_flag
            FROM targets
            WHERE loan_id = 'L6'
            """
        ).fetchone() == (False,)
        assert con.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN cure_date IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN redefault_flag THEN 1 ELSE 0 END)
            FROM events
            WHERE loan_id = 'L4'
            """
        ).fetchone() == (2, 1, 1)


def test_future_changes_move_targets_but_not_historical_observations(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_gold_fixture(repo, future_defaults=True)
    run_target_build(all_targets=True, force=True, repo_root=repo)

    target_path = repo / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
    target_glob = f"{target_path.as_posix()}/*.parquet"
    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"CREATE VIEW targets AS SELECT * FROM read_parquet('{target_glob}')"
        )
        baseline = con.execute(
            """
            SELECT default_due_to_delinquency, default_next_12m
            FROM targets
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-01-01'
            """
        ).fetchone()

    shutil.rmtree(repo / "data" / "gold" / "freddie" / "targets")
    shutil.rmtree(repo / "data" / "gold" / "freddie" / "loan_month")
    shutil.rmtree(repo / "data" / "gold" / "freddie" / "loan_static")
    _write_gold_fixture(repo, future_defaults=False)
    run_target_build(all_targets=True, force=True, repo_root=repo)

    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"CREATE VIEW targets AS SELECT * FROM read_parquet('{target_glob}')"
        )
        changed = con.execute(
            """
            SELECT default_due_to_delinquency, default_next_12m
            FROM targets
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-01-01'
            """
        ).fetchone()

    assert baseline == (False, 1)
    assert changed == (False, 0)
