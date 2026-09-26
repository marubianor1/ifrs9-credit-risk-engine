"""Build Gold point-in-time loan-month mart partitions."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.feature_registry import load_default_registry
from ifrs9.mart.loan_static import loan_static_sql
from ifrs9.mart.validation import validate_gold_with_duckdb
from ifrs9.mart.views import register_gold_views

LOGGER = logging.getLogger(__name__)


class MartBuildError(RuntimeError):
    """Raised when point-in-time mart construction fails."""


@dataclass(frozen=True)
class MartPartitionResult:
    """Manifest result for one mart vintage."""

    year: int
    loan_static_rows: int
    loan_month_rows: int
    started_at: str
    completed_at: str
    status: str
    processing_time_seconds: float


def configure_logging() -> None:
    """Configure command-line logging for mart builds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "data.yaml").open() as stream:
        return yaml.safe_load(stream)


def _discover_years(silver_root: Path) -> list[int]:
    return sorted(
        int(path.name.split("=", 1)[1])
        for path in (silver_root / "origination").glob("vintage_year=*")
        if path.is_dir()
    )


def _partition_path(root: Path, table: str, year: int) -> Path:
    return root / table / f"vintage_year={year}" / f"part-{table}.parquet"


def _prepare_tmp(final_path: Path) -> tuple[Path, Path]:
    final_dir = final_path.parent
    tmp_dir = final_dir.parent / f".{final_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir, tmp_dir / final_path.name


def _promote_tmp(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def _copy_query(con: duckdb.DuckDBPyConnection, query: str, output_path: Path) -> None:
    con.execute(
        f"""
        COPY ({query})
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def loan_month_sql(perf_path: Path, static_path: Path, processed_at: str) -> str:
    """Return SQL for one Gold loan_month partition."""
    return f"""
    WITH p AS (
        SELECT * FROM read_parquet('{perf_path.as_posix()}')
    ),
    s AS (
        SELECT loan_id, vintage_year, origination_date, first_payment_date, maturity_date,
               original_upb, original_interest_rate
        FROM read_parquet('{static_path.as_posix()}')
    ),
    base AS (
        SELECT
            p.loan_id,
            p._source_year AS vintage_year,
            p.period AS as_of_date,
            p.loan_age AS loan_age_source,
            p.remaining_months_to_legal_maturity AS remaining_months_to_maturity_source,
            p.current_actual_upb,
            p.current_interest_rate,
            p.delinquency_months,
            p.delinquency_status_type AS delinquency_status_category,
            p.current_loan_delinquency_status AS delinquency_status_raw,
            p.modification_flag,
            p.payment_deferral_flag,
            p.borrower_assistance_status_code,
            p.delinquency_due_to_disaster,
            p.current_non_interest_bearing_upb,
            p.current_interest_bearing_upb,
            p.zero_balance_code,
            p.zero_balance_effective_date,
            p.defect_settlement_date,
            p.due_date_of_last_paid_installment,
            p.mi_recoveries,
            p.net_sale_proceeds,
            p.non_mi_recoveries,
            p.total_expenses,
            p.legal_costs,
            p.maintenance_and_preservation_costs,
            p.taxes_and_insurance,
            p.miscellaneous_expenses,
            p.actual_loss,
            p.cumulative_modification_costs,
            p.current_period_modification_costs,
            p.bankruptcy_cramdown_costs,
            p.estimated_loan_to_value,
            p.mortgage_insurance_cancellation_indicator,
            p.servicer_name,
            s.origination_date,
            s.first_payment_date,
            s.maturity_date,
            s.original_upb,
            s.original_interest_rate,
            p._source_zip,
            p._source_member,
            p._ingested_at,
            p._schema_version,
            p._silver_processed_at
        FROM p
        LEFT JOIN s
          ON p.loan_id = s.loan_id
         AND p._source_year = s.vintage_year
    ),
    lags AS (
        SELECT
            b.*,
            l1.delinquency_months AS delinquency_months_lag_1,
            l3.delinquency_months AS delinquency_months_lag_3,
            l6.delinquency_months AS delinquency_months_lag_6,
            l12.delinquency_months AS delinquency_months_lag_12,
            l1.current_actual_upb AS current_upb_lag_1,
            l3.current_actual_upb AS current_upb_lag_3,
            l6.current_actual_upb AS current_upb_lag_6,
            l12.current_actual_upb AS current_upb_lag_12,
            l1.current_interest_rate AS current_interest_rate_lag_1,
            l12.current_interest_rate AS current_interest_rate_lag_12
        FROM base b
        LEFT JOIN base l1
          ON b.loan_id = l1.loan_id
         AND b.as_of_date = l1.as_of_date + INTERVAL 1 MONTH
        LEFT JOIN base l3
          ON b.loan_id = l3.loan_id
         AND b.as_of_date = l3.as_of_date + INTERVAL 3 MONTH
        LEFT JOIN base l6
          ON b.loan_id = l6.loan_id
         AND b.as_of_date = l6.as_of_date + INTERVAL 6 MONTH
        LEFT JOIN base l12
          ON b.loan_id = l12.loan_id
         AND b.as_of_date = l12.as_of_date + INTERVAL 12 MONTH
    ),
    with_prior_observation AS (
        SELECT
            *,
            lag(as_of_date) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
            ) AS prior_observation_as_of_date
        FROM lags
    ),
    gap_flags AS (
        SELECT
            *,
            CASE
                WHEN prior_observation_as_of_date IS NULL THEN 0
                WHEN date_diff('month', prior_observation_as_of_date, as_of_date) > 1 THEN 1
                ELSE 0
            END AS current_reporting_gap_flag
        FROM with_prior_observation
    ),
    hist AS (
        SELECT
            *,
            row_number() OVER (PARTITION BY loan_id ORDER BY as_of_date) AS observed_months_to_date,
            date_diff(
                'month',
                min(as_of_date) OVER (PARTITION BY loan_id),
                as_of_date
            ) + 1 AS calendar_months_since_first_observation,
            max(delinquency_months) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS max_delinquency_months_to_date,
            sum(CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS months_delinquent_to_date,
            max(CASE WHEN delinquency_months > 0 THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS last_delinquent_as_of_date,
            sum(CASE WHEN delinquency_months >= 1 THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS months_1plus_delinquent_to_date,
            sum(CASE WHEN delinquency_months >= 2 THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS months_2plus_delinquent_to_date,
            sum(CASE WHEN delinquency_months >= 3 THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS months_3plus_delinquent_to_date,
            sum(CASE WHEN modification_flag = 'Y' THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) > 0 AS ever_modified_to_date,
            max(CASE WHEN modification_flag = 'Y' THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS last_modification_as_of_date,
            sum(CASE WHEN borrower_assistance_status_code IS NOT NULL
                       OR payment_deferral_flag IS NOT NULL
                     THEN 1 ELSE 0 END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) > 0 AS ever_assistance_to_date,
            max(current_reporting_gap_flag) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) > 0 AS has_reporting_gap_to_date
        FROM gap_flags
    )
    SELECT
        loan_id,
        vintage_year,
        as_of_date,
        loan_age_source,
        date_diff('month', first_payment_date, as_of_date) + 1 AS months_on_book_derived,
        date_diff('month', origination_date, as_of_date) AS months_since_origination,
        remaining_months_to_maturity_source,
        date_diff('month', as_of_date, maturity_date) AS months_to_contractual_maturity,
        current_actual_upb,
        current_interest_rate,
        delinquency_months,
        delinquency_status_category,
        delinquency_status_raw,
        modification_flag,
        payment_deferral_flag,
        borrower_assistance_status_code,
        delinquency_due_to_disaster,
        current_non_interest_bearing_upb,
        current_interest_bearing_upb,
        zero_balance_code,
        zero_balance_effective_date,
        defect_settlement_date,
        due_date_of_last_paid_installment,
        estimated_loan_to_value,
        mortgage_insurance_cancellation_indicator,
        servicer_name,
        CASE WHEN original_upb > 0 THEN current_actual_upb / original_upb ELSE NULL END
            AS current_upb_to_original_upb,
        CASE WHEN original_upb IS NOT NULL THEN original_upb - current_actual_upb ELSE NULL END
            AS principal_reduction_to_date,
        current_interest_rate - original_interest_rate AS interest_rate_change_from_origination,
        delinquency_months_lag_1,
        delinquency_months_lag_3,
        delinquency_months_lag_6,
        delinquency_months_lag_12,
        current_upb_lag_1,
        current_upb_lag_3,
        current_upb_lag_6,
        current_upb_lag_12,
        current_interest_rate_lag_1,
        current_interest_rate_lag_12,
        current_actual_upb - current_upb_lag_1 AS upb_change_1m,
        CASE WHEN current_upb_lag_1 > 0
             THEN (current_actual_upb - current_upb_lag_1) / current_upb_lag_1
             ELSE NULL END AS upb_change_pct_1m,
        current_actual_upb - current_upb_lag_3 AS upb_change_3m,
        CASE WHEN current_upb_lag_3 > 0
             THEN (current_actual_upb - current_upb_lag_3) / current_upb_lag_3
             ELSE NULL END AS upb_change_pct_3m,
        current_actual_upb - current_upb_lag_6 AS upb_change_6m,
        CASE WHEN current_upb_lag_6 > 0
             THEN (current_actual_upb - current_upb_lag_6) / current_upb_lag_6
             ELSE NULL END AS upb_change_pct_6m,
        current_actual_upb - current_upb_lag_12 AS upb_change_12m,
        CASE WHEN current_upb_lag_12 > 0
             THEN (current_actual_upb - current_upb_lag_12) / current_upb_lag_12
             ELSE NULL END AS upb_change_pct_12m,
        current_interest_rate - current_interest_rate_lag_1 AS rate_change_1m,
        current_interest_rate - current_interest_rate_lag_12 AS rate_change_12m,
        max_delinquency_months_to_date,
        months_delinquent_to_date,
        CASE WHEN delinquency_months_lag_1 = 0 AND delinquency_months > 0
             THEN true ELSE false END AS delinquency_event_current_month,
        sum(CASE WHEN delinquency_months_lag_1 = 0 AND delinquency_months > 0
                 THEN 1 ELSE 0 END) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS delinquency_events_to_date,
        CASE WHEN last_delinquent_as_of_date IS NULL THEN NULL
             ELSE date_diff('month', last_delinquent_as_of_date, as_of_date)
        END AS months_since_last_delinquency,
        max(delinquency_months) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW
        ) AS max_delinquency_last_3m,
        max(delinquency_months) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 5 MONTH PRECEDING AND CURRENT ROW
        ) AS max_delinquency_last_6m,
        max(delinquency_months) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 11 MONTH PRECEDING AND CURRENT ROW
        ) AS max_delinquency_last_12m,
        sum(CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW
        ) AS months_delinquent_last_3m,
        sum(CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 5 MONTH PRECEDING AND CURRENT ROW
        ) AS months_delinquent_last_6m,
        sum(CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY loan_id ORDER BY as_of_date
            RANGE BETWEEN INTERVAL 11 MONTH PRECEDING AND CURRENT ROW
        ) AS months_delinquent_last_12m,
        months_1plus_delinquent_to_date,
        months_2plus_delinquent_to_date,
        months_3plus_delinquent_to_date,
        ever_modified_to_date,
        CASE WHEN last_modification_as_of_date IS NULL THEN NULL
             ELSE date_diff('month', last_modification_as_of_date, as_of_date)
        END AS months_since_last_modification,
        ever_assistance_to_date,
        (borrower_assistance_status_code IS NOT NULL OR payment_deferral_flag IS NOT NULL)
            AS current_assistance_flag,
        observed_months_to_date,
        calendar_months_since_first_observation,
        coalesce(has_reporting_gap_to_date, false) AS has_reporting_gap_to_date,
        mi_recoveries,
        net_sale_proceeds,
        non_mi_recoveries,
        total_expenses,
        legal_costs,
        maintenance_and_preservation_costs,
        taxes_and_insurance,
        miscellaneous_expenses,
        actual_loss,
        cumulative_modification_costs,
        current_period_modification_costs,
        bankruptcy_cramdown_costs,
        _source_zip,
        _source_member,
        _ingested_at,
        _schema_version,
        _silver_processed_at,
        '{processed_at}' AS _gold_processed_at
    FROM hist
    """

def _git_state(repo_root: Path) -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}
    return {"commit": commit, "dirty": str(bool(status))}


def _build_year(
    con: duckdb.DuckDBPyConnection,
    year: int,
    silver_root: Path,
    gold_root: Path,
    force: bool,
) -> MartPartitionResult:
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    static_final = _partition_path(gold_root, "loan_static", year)
    month_final = _partition_path(gold_root, "loan_month", year)
    if static_final.exists() and month_final.exists() and not force:
        static_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [static_final.as_posix()]
        ).fetchone()[0]
        month_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [month_final.as_posix()]
        ).fetchone()[0]
        return MartPartitionResult(
            year, static_rows, month_rows, started_at, datetime.now(UTC).isoformat(),
            "skipped_existing", time.perf_counter() - start
        )

    processed_at = datetime.now(UTC).isoformat()
    silver_static = _partition_path(silver_root, "origination", year)
    silver_month = _partition_path(silver_root, "performance", year)
    static_tmp_dir, static_tmp = _prepare_tmp(static_final)
    month_tmp_dir, month_tmp = _prepare_tmp(month_final)
    try:
        _copy_query(con, loan_static_sql(silver_static, processed_at), static_tmp)
        _copy_query(con, loan_month_sql(silver_month, static_tmp, processed_at), month_tmp)
        _promote_tmp(static_tmp_dir, static_final.parent)
        _promote_tmp(month_tmp_dir, month_final.parent)
    except Exception:
        for tmp in [static_tmp_dir, month_tmp_dir]:
            if tmp.exists():
                shutil.rmtree(tmp)
        raise
    static_rows = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [static_final.as_posix()]
    ).fetchone()[0]
    month_rows = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [month_final.as_posix()]
    ).fetchone()[0]
    return MartPartitionResult(
        year, static_rows, month_rows, started_at, datetime.now(UTC).isoformat(),
        "success", time.perf_counter() - start
    )


def _write_manifest(
    manifest_dir: Path,
    results: list[MartPartitionResult],
    registry_version: str,
    git_state: dict[str, str],
    elapsed: float,
    validation: dict[str, object],
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_silver_version": "v1",
        "years_processed": [result.year for result in results],
        "loan_static_rows": validation["loan_static_rows"],
        "loan_month_rows": validation["loan_month_rows"],
        "distinct_monthly_loans": validation["loan_month_distinct_loans"],
        "feature_registry_version": registry_version,
        "build_timestamp": datetime.now(UTC).isoformat(),
        "processing_time": elapsed,
        "status": "success",
        "git": git_state,
        "partitions": [asdict(result) for result in results],
    }
    (manifest_dir / "mart_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_profile(con: duckdb.DuckDBPyConnection, manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
          WITH loan_counts AS (
            SELECT vintage_year, loan_id, COUNT(*) AS observed_months,
                   MAX(CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END) AS ever_dq,
                   MAX(CASE WHEN ever_modified_to_date THEN 1 ELSE 0 END) AS ever_mod,
                   MAX(CASE WHEN has_reporting_gap_to_date THEN 1 ELSE 0 END) AS has_gap
            FROM gold_loan_month
            GROUP BY 1,2
          ),
          vintage_bounds AS (
            SELECT vintage_year,
                   MIN(as_of_date) AS min_as_of_date,
                   MAX(as_of_date) AS max_as_of_date
            FROM gold_loan_month
            GROUP BY 1
          )
          SELECT lc.vintage_year,
                 SUM(lc.observed_months) AS rows,
                 COUNT(*) AS unique_loans,
                 vb.min_as_of_date,
                 vb.max_as_of_date,
                 AVG(lc.observed_months) AS avg_observed_months_per_loan,
                 median(lc.observed_months) AS median_observed_months_per_loan,
                 MAX(lc.observed_months) AS max_observed_months_per_loan,
                 SUM(lc.ever_dq) AS loans_ever_delinquent,
                 SUM(lc.ever_mod) AS loans_ever_modified,
                 SUM(lc.has_gap) AS loans_with_reporting_gaps
          FROM loan_counts lc
          JOIN vintage_bounds vb USING (vintage_year)
          GROUP BY lc.vintage_year, vb.min_as_of_date, vb.max_as_of_date
          ORDER BY lc.vintage_year
        )
        TO '{(manifest_dir / "loan_month_profile.csv").as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )


def _write_feature_population(con: duckdb.DuckDBPyConnection, manifest_dir: Path) -> None:
    features = [
        "months_on_book_derived",
        "current_upb_to_original_upb",
        "max_delinquency_months_to_date",
        "months_since_last_delinquency",
        "observed_months_to_date",
        "delinquency_months_lag_1",
        "current_upb_lag_12",
    ]
    rows = []
    for feature in features:
        result = con.execute(
            f"""
            SELECT vintage_year, COUNT({feature}), COUNT(*)
            FROM gold_loan_month
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        rows.extend(
            {
                "feature": feature,
                "year": year,
                "non_null_count": non_null,
                "non_null_pct": non_null / total if total else None,
            }
            for year, non_null, total in result
        )
    with (manifest_dir / "feature_population.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution_profile(con: duckdb.DuckDBPyConnection, manifest_dir: Path) -> None:
    features = [
        "months_on_book_derived",
        "current_upb_to_original_upb",
        "max_delinquency_months_to_date",
        "months_since_last_delinquency",
        "observed_months_to_date",
    ]
    rows = []
    for feature in features:
        row = con.execute(
            f"""
            SELECT '{feature}' AS feature,
                   MIN({feature}), MAX({feature}), AVG({feature}), median({feature}),
                   quantile_cont({feature}, 0.01),
                   quantile_cont({feature}, 0.99),
                   SUM(CASE WHEN {feature} IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
            FROM gold_loan_month
            """
        ).fetchone()
        rows.append(
            {
                "feature": row[0],
                "min": row[1],
                "max": row[2],
                "mean": row[3],
                "median": row[4],
                "p01": row[5],
                "p99": row[6],
                "null_rate": row[7],
            }
        )
    with (manifest_dir / "feature_distribution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_point_in_time_mart_build(
    *,
    year: int | None = None,
    all_years: bool = False,
    force: bool = False,
    repo_root: Path | None = None,
) -> list[MartPartitionResult]:
    """Build Gold point-in-time loan static and loan-month partitions."""
    configure_logging()
    root = find_repo_root(repo_root)
    config = _load_config(root)
    silver_root = root / config["ingestion"]["silver_freddie_dir"]
    gold_root = root / config["ingestion"]["gold_freddie_dir"]
    manifest_dir = root / config["ingestion"]["mart_manifest_dir"]
    available_years = _discover_years(silver_root)
    years = available_years if all_years else [year]
    if not years or years == [None]:
        msg = "Specify a year or all_years=True"
        raise MartBuildError(msg)
    missing = [selected for selected in years if selected not in available_years]
    if missing:
        msg = f"Missing Silver partitions for years: {missing}"
        raise MartBuildError(msg)
    registry = load_default_registry(root)
    start = time.perf_counter()
    results = []
    with duckdb.connect(database=":memory:") as con:
        con.execute("PRAGMA threads=4")
        for selected_year in years:
            LOGGER.info("Building Gold mart vintage %s", selected_year)
            results.append(_build_year(con, selected_year, silver_root, gold_root, force))
        register_gold_views(con, root)
        _write_profile(con, manifest_dir)
        _write_feature_population(con, manifest_dir)
        _write_distribution_profile(con, manifest_dir)
    validation = validate_gold_with_duckdb(root)
    _write_manifest(
        manifest_dir,
        results,
        registry.version,
        _git_state(root),
        time.perf_counter() - start,
        validation,
    )
    LOGGER.info("Point-in-time mart build completed")
    return results
