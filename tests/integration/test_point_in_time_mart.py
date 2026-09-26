from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from ifrs9.mart.loan_month import run_point_in_time_mart_build
from ifrs9.mart.validation import validate_gold_with_duckdb
from ifrs9.mart.views import register_gold_views


def _repo_fixture(tmp_path: Path) -> Path:
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
                "  silver_freddie_dir: data/silver/freddie",
                "  gold_freddie_dir: data/gold/freddie",
                "  mart_manifest_dir: artifacts/mart",
                "",
            ]
        )
    )
    return repo


def _write_silver_fixture(
    repo: Path,
    *,
    future_actual_loss: float = 700.0,
    include_future: bool = True,
) -> None:
    silver = repo / "data" / "silver" / "freddie"
    origination = silver / "origination" / "vintage_year=2026"
    performance = silver / "performance" / "vintage_year=2026"
    origination.mkdir(parents=True)
    performance.mkdir(parents=True)
    future_row = (
        f"""
                ('L1', 2026, DATE '2026-04-01', 3, 356, 96000.0, 6.25, 0, 'current', '0', NULL,
                 NULL, 'A', NULL, 0.0, 96000.0, NULL, NULL, NULL, NULL, NULL, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {future_actual_loss}, 0.0, 0.0, 0.0, 75.0,
                 NULL, 'SERVICER', 'zip', 'perf.txt', TIMESTAMP '2026-01-02 00:00:00', 'v1',
                 TIMESTAMP '2026-01-03 00:00:00'),"""
        if include_future
        else ""
    )

    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM (
                VALUES
                ('L1', 2026, DATE '2026-01-01', DATE '2026-02-01', DATE '2056-01-01',
                 740, NULL, 'N', 32.0, 75.0, 75.0, 100000.0, 6.0, 360, 1, 1, 'P',
                 'SF', 'P', 'R', 0.0, 'CA', '90001', 'AVM', 'VALID', 'N', 'SELLER',
                 NULL, 'valid', NULL, 'valid', 'valid', 'valid', 'zip', 'orig.txt',
                 TIMESTAMP '2026-01-02 00:00:00', 'v1', TIMESTAMP '2026-01-03 00:00:00'),
                ('L2', 2026, DATE '2026-01-01', DATE '2026-02-01', DATE '2056-01-01',
                 700, NULL, 'N', 36.0, 80.0, 80.0, 120000.0, 6.5, 360, 2, 1, 'P',
                 'SF', 'P', 'R', 0.0, 'CA', '90002', 'AVM', 'VALID', 'N', 'SELLER',
                 NULL, 'valid', NULL, 'valid', 'valid', 'valid', 'zip', 'orig.txt',
                 TIMESTAMP '2026-01-02 00:00:00', 'v1', TIMESTAMP '2026-01-03 00:00:00')
              ) AS t(
                loan_id, _source_year, origination_month, first_payment_date, maturity_date,
                credit_score, vantagescore_4, first_time_homebuyer_flag,
                original_debt_to_income_ratio, original_loan_to_value,
                original_combined_loan_to_value, original_upb, original_interest_rate,
                original_loan_term, number_of_borrowers, number_of_units, occupancy_status,
                property_type, loan_purpose, channel, mortgage_insurance_percentage,
                property_state, postal_code, property_valuation_method,
                property_valuation_method_status, interest_only_indicator, seller_name,
                special_eligibility_program, credit_score_status, vantagescore_4_status,
                original_debt_to_income_ratio_status, original_loan_to_value_status,
                original_combined_loan_to_value_status, _source_zip, _source_member,
                _ingested_at, _schema_version, _silver_processed_at
              )
            )
            TO '{(origination / "part-origination.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM (
                VALUES
                ('L1', 2026, DATE '2026-01-01', 0, 359, 99000.0, 6.0, 0, 'current', '0', NULL,
                 NULL, NULL, NULL, 0.0, 99000.0, NULL, NULL, NULL, NULL, NULL, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 78.0,
                 NULL, 'SERVICER', 'zip',
                 'perf.txt', TIMESTAMP '2026-01-02 00:00:00', 'v1',
                 TIMESTAMP '2026-01-03 00:00:00'),
                ('L1', 2026, DATE '2026-03-01', 2, 357, 97000.0, 6.0, 2, 'delinquent', '2', 'Y',
                 NULL, NULL, NULL, 0.0, 97000.0, NULL, NULL, NULL, NULL, NULL, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 76.0,
                 NULL, 'SERVICER', 'zip',
                 'perf.txt', TIMESTAMP '2026-01-02 00:00:00', 'v1',
                 TIMESTAMP '2026-01-03 00:00:00'),
{future_row}
                ('L2', 2026, DATE '2026-02-01', 1, 358, 119000.0, 6.5, 0, 'current', '0', NULL,
                 NULL, NULL, NULL, 0.0, 119000.0, NULL, NULL, NULL, NULL, NULL, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 79.0,
                 NULL, 'SERVICER', 'zip',
                 'perf.txt', TIMESTAMP '2026-01-02 00:00:00', 'v1',
                 TIMESTAMP '2026-01-03 00:00:00')
              ) AS t(
                loan_id, _source_year, period, loan_age, remaining_months_to_legal_maturity,
                current_actual_upb, current_interest_rate, delinquency_months,
                delinquency_status_type,
                current_loan_delinquency_status, modification_flag, payment_deferral_flag,
                borrower_assistance_status_code, delinquency_due_to_disaster,
                current_non_interest_bearing_upb, current_interest_bearing_upb,
                zero_balance_code, zero_balance_effective_date, defect_settlement_date,
                due_date_of_last_paid_installment, mi_recoveries, net_sale_proceeds,
                non_mi_recoveries, total_expenses, legal_costs,
                maintenance_and_preservation_costs, taxes_and_insurance,
                miscellaneous_expenses, actual_loss, cumulative_modification_costs,
                current_period_modification_costs, bankruptcy_cramdown_costs,
                estimated_loan_to_value, mortgage_insurance_cancellation_indicator,
                servicer_name, _source_zip, _source_member, _ingested_at, _schema_version,
                _silver_processed_at
              )
            )
            TO '{(performance / "part-performance.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )


def test_point_in_time_mart_builds_views_calendar_gaps_and_artifacts(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_silver_fixture(repo)

    result = run_point_in_time_mart_build(year=2026, force=True, repo_root=repo)

    assert result[0].loan_static_rows == 2
    assert result[0].loan_month_rows == 4
    validation = validate_gold_with_duckdb(repo)
    assert validation["primary_key_duplicate_count"] == 0
    assert validation["cross_vintage_duplicate_loan_ids"] == 0
    assert validation["analytical_view_rows"] == 4

    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo)
        row = con.execute(
            """
            SELECT delinquency_months_lag_1, has_reporting_gap_to_date,
                   max_delinquency_months_to_date, months_since_last_delinquency,
                   months_since_last_modification
            FROM gold_loan_month
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone()
        assert row == (None, True, 2, 0, 0)
        assert con.execute("SELECT COUNT(*) FROM gold_loan_month_analytical").fetchone()[0] == 4

    manifest = json.loads((repo / "artifacts" / "mart" / "mart_manifest.json").read_text())
    assert manifest["loan_month_rows"] == 4
    assert (repo / "artifacts" / "mart" / "loan_month_profile.csv").exists()
    assert (repo / "artifacts" / "mart" / "feature_population.csv").exists()
    assert (repo / "artifacts" / "mart" / "feature_distribution.csv").exists()


def test_future_perturbation_does_not_change_prior_safe_features(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_silver_fixture(repo, future_actual_loss=700.0)
    run_point_in_time_mart_build(year=2026, force=True, repo_root=repo)

    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo)
        baseline = con.execute(
            """
            SELECT current_upb_to_original_upb, max_delinquency_months_to_date,
                   months_delinquent_to_date, delinquency_months_lag_1,
                   current_upb_lag_1, actual_loss
            FROM gold_loan_month
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone()

    shutil.rmtree(repo / "data" / "gold")
    shutil.rmtree(repo / "artifacts" / "mart")
    shutil.rmtree(repo / "data" / "silver")
    _write_silver_fixture(repo, future_actual_loss=999999.0)
    run_point_in_time_mart_build(year=2026, force=True, repo_root=repo)

    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo)
        perturbed = con.execute(
            """
            SELECT current_upb_to_original_upb, max_delinquency_months_to_date,
                   months_delinquent_to_date, delinquency_months_lag_1,
                   current_upb_lag_1, actual_loss
            FROM gold_loan_month
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone()

    assert perturbed == baseline


def test_future_truncation_does_not_change_prior_safe_features(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_silver_fixture(repo, include_future=True)
    run_point_in_time_mart_build(year=2026, force=True, repo_root=repo)

    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo)
        baseline = con.execute(
            """
            SELECT max_delinquency_months_to_date, months_delinquent_to_date,
                   delinquency_events_to_date, ever_assistance_to_date,
                   has_reporting_gap_to_date
            FROM gold_loan_month
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone()

    shutil.rmtree(repo / "data" / "gold")
    shutil.rmtree(repo / "artifacts" / "mart")
    shutil.rmtree(repo / "data" / "silver")
    _write_silver_fixture(repo, include_future=False)
    run_point_in_time_mart_build(year=2026, force=True, repo_root=repo)

    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo)
        truncated = con.execute(
            """
            SELECT max_delinquency_months_to_date, months_delinquent_to_date,
                   delinquency_events_to_date, ever_assistance_to_date,
                   has_reporting_gap_to_date
            FROM gold_loan_month
            WHERE loan_id = 'L1' AND as_of_date = DATE '2026-03-01'
            """
        ).fetchone()

    assert truncated == baseline
