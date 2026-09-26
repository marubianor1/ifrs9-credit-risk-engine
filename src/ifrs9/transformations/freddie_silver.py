"""Build Freddie Mac Silver standardized Parquet partitions from Bronze."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import yaml

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.transformations.missing_values import (
    SENTINEL_RULES,
    nullify_sentinels,
    status_expression,
)
from ifrs9.transformations.normalization import (
    delinquency_months_expr,
    delinquency_status_type_expr,
    yyyymm_to_month_date,
)
from ifrs9.transformations.validation import SilverDQMetrics, origination_dq, performance_dq

LOGGER = logging.getLogger(__name__)


class SilverBuildError(RuntimeError):
    """Raised when Silver transformation fails."""


@dataclass(frozen=True)
class SilverPartitionResult:
    """Manifest entry for a transformed Silver partition."""

    year: int
    dataset: str
    bronze_input_rows: int
    silver_output_rows: int
    schema_version: str
    started_at: str
    completed_at: str
    status: str
    warnings: list[str]


def configure_logging() -> None:
    """Configure readable structured logging for Silver build commands."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "data.yaml").open() as stream:
        return yaml.safe_load(stream)


def _discover_years(bronze_root: Path) -> list[int]:
    return sorted(
        int(path.name.split("=", 1)[1])
        for path in (bronze_root / "origination").glob("vintage_year=*")
        if path.is_dir()
    )


def _partition_path(root: Path, dataset: str, year: int) -> Path:
    return root / dataset / f"vintage_year={year}" / f"part-{dataset}.parquet"


def _write_atomic(frame: pl.DataFrame, final_path: Path, force: bool) -> Path:
    if final_path.exists() and not force:
        return final_path
    final_dir = final_path.parent
    tmp_dir = final_dir.parent / f".{final_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    tmp_file = tmp_dir / final_path.name
    try:
        frame.write_parquet(tmp_file, compression="zstd")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return final_path


def _date_columns(frame: pl.DataFrame, columns: list[str]) -> list[pl.Expr]:
    return [yyyymm_to_month_date(column) for column in columns if column in frame.columns]


def _origination_month_expr() -> pl.Expr:
    raw = pl.col("loan_id").str.extract(r"^[A-Z](\d{2})Q([1-4])", 1)
    quarter = pl.col("loan_id").str.extract(r"^[A-Z](\d{2})Q([1-4])", 2)
    month = (
        pl.when(quarter == "1")
        .then(pl.lit("01"))
        .when(quarter == "2")
        .then(pl.lit("04"))
        .when(quarter == "3")
        .then(pl.lit("07"))
        .when(quarter == "4")
        .then(pl.lit("10"))
        .otherwise(None)
    )
    yyyymmdd = pl.when(raw.is_not_null()).then(pl.lit("20") + raw + month + pl.lit("01"))
    return yyyymmdd.str.strptime(pl.Date, "%Y%m%d", strict=False).alias("origination_month")


def transform_origination(frame: pl.DataFrame) -> pl.DataFrame:
    """Transform one Bronze origination partition into Silver."""
    status_fields = [
        "credit_score",
        "vantagescore_4",
        "original_debt_to_income_ratio",
        "original_loan_to_value",
        "original_combined_loan_to_value",
        "property_valuation_method",
    ]
    transformed = frame.with_columns(
        [status_expression(field).alias(f"{field}_status") for field in status_fields]
    ).with_columns([nullify_sentinels(field) for field in status_fields])
    transformed = transformed.with_columns(
        _date_columns(transformed, ["first_payment_date", "maturity_date"])
        + [_origination_month_expr()]
    )
    return transformed.with_columns(
        pl.lit(datetime.now(UTC).isoformat()).alias("_silver_processed_at")
    )


def transform_performance(frame: pl.DataFrame) -> pl.DataFrame:
    """Transform one Bronze performance partition into Silver."""
    status_fields = [
        "estimated_loan_to_value",
        "net_sale_proceeds",
        "mortgage_insurance_cancellation_indicator",
    ]
    transformed = frame.with_columns(
        [status_expression(field).alias(f"{field}_status") for field in status_fields]
    ).with_columns([nullify_sentinels(field) for field in status_fields])
    transformed = transformed.with_columns(
        _date_columns(
            transformed,
            [
                "period",
                "defect_settlement_date",
                "zero_balance_effective_date",
                "due_date_of_last_paid_installment",
            ],
        )
        + [delinquency_months_expr(), delinquency_status_type_expr()]
    )
    return transformed.with_columns(
        pl.lit(datetime.now(UTC).isoformat()).alias("_silver_processed_at")
    )


def _normalization_rows(frame: pl.DataFrame, dataset: str, year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    applicable = [field for field in SENTINEL_RULES if field in frame.columns]
    for field in applicable:
        status_col = f"{field}_status"
        counts = frame.group_by(status_col).len().to_dicts()
        by_status = {row[status_col]: row["len"] for row in counts}
        source_rows = frame.height
        rows.append(
            {
                "dataset": dataset,
                "year": year,
                "field": field,
                "source_rows": source_rows,
                "valid_values": by_status.get("VALID_VALUE", 0),
                "normalized_nulls": source_rows - by_status.get("VALID_VALUE", 0),
                "sentinel_values": source_rows - by_status.get("VALID_VALUE", 0),
                "invalid_values": by_status.get("INVALID", 0),
                "conversion_failures": 0,
                "structural_missing": by_status.get("STRUCTURAL_NOT_AVAILABLE", 0),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, results: list[SilverPartitionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_bronze(path: Path) -> pl.DataFrame:
    if not path.exists():
        msg = f"Bronze input not found: {path}"
        raise SilverBuildError(msg)
    return pl.read_parquet(path)


def _process_year(
    year: int,
    bronze_root: Path,
    silver_root: Path,
    force: bool,
) -> tuple[list[SilverPartitionResult], list[SilverDQMetrics], list[dict[str, Any]]]:
    LOGGER.info("Building Silver vintage %s", year)
    results: list[SilverPartitionResult] = []
    dq: list[SilverDQMetrics] = []
    normalization: list[dict[str, Any]] = []
    for dataset, transform, checks in [
        ("origination", transform_origination, origination_dq),
        ("performance", transform_performance, performance_dq),
    ]:
        started_at = datetime.now(UTC).isoformat()
        bronze_path = _partition_path(bronze_root, dataset, year)
        silver_path = _partition_path(silver_root, dataset, year)
        if silver_path.exists() and not force:
            LOGGER.info("Silver %s %s exists; skipping without --force", dataset, year)
            row_count = pl.scan_parquet(silver_path).select(pl.len()).collect().item()
            results.append(
                SilverPartitionResult(
                    year,
                    dataset,
                    row_count,
                    row_count,
                    "v1",
                    started_at,
                    datetime.now(UTC).isoformat(),
                    "skipped_existing",
                    [],
                )
            )
            continue
        bronze = _load_bronze(bronze_path)
        silver = transform(bronze)
        _write_atomic(silver, silver_path, force=force)
        if bronze.height != silver.height:
            msg = f"Row count changed for {dataset} {year}: {bronze.height} -> {silver.height}"
            raise SilverBuildError(msg)
        dq.extend(checks(silver, year))
        normalization.extend(_normalization_rows(silver, dataset, year))
        results.append(
            SilverPartitionResult(
                year,
                dataset,
                bronze.height,
                silver.height,
                "v1",
                started_at,
                datetime.now(UTC).isoformat(),
                "success",
                [],
            )
        )
    return results, dq, normalization


def _historical_availability(con: duckdb.DuckDBPyConnection, output: Path) -> None:
    rows = []
    for dataset in ["origination", "performance"]:
        columns = con.execute(
            "DESCRIBE SELECT * FROM "
            f"read_parquet('data/silver/freddie/{dataset}/*/*.parquet') LIMIT 0"
        ).fetchall()
        fields = [
            name
            for name, *_ in columns
            if not name.startswith("_") and not name.endswith("_status")
        ]
        for field in fields:
            result = con.execute(
                f"""
                SELECT _source_year AS year,
                       ? AS field,
                       COUNT(*) AS row_count,
                       COUNT({field}) AS non_null_count,
                       COUNT({field}) * 1.0 / COUNT(*) AS non_null_pct,
                       COUNT(DISTINCT {field}) AS distinct_count
                FROM read_parquet('data/silver/freddie/{dataset}/*/*.parquet')
                GROUP BY 1 ORDER BY 1
                """,
                [field],
            ).fetchall()
            for row in result:
                rows.append(
                    {
                        "dataset": dataset,
                        "year": row[0],
                        "field": row[1],
                        "row_count": row[2],
                        "non_null_count": row[3],
                        "non_null_pct": row[4],
                        "distinct_count": row[5],
                        "sentinel_count": "",
                        "sentinel_pct": "",
                    }
                )
    _write_csv(output, rows)


def _temporal_gap_report(con: duckdb.DuckDBPyConnection, output: Path) -> dict[str, int]:
    query = """
    WITH perf AS (
      SELECT loan_id, period,
             date_diff(
               'month',
               lag(period) OVER (PARTITION BY loan_id ORDER BY period),
               period
             ) AS gap
      FROM read_parquet('data/silver/freddie/performance/*/*.parquet')
    ),
    gaps AS (
      SELECT loan_id, gap - 1 AS missing_months
      FROM perf
      WHERE gap > 1
    )
    SELECT COUNT(DISTINCT loan_id),
           COALESCE(SUM(missing_months), 0),
           COALESCE(MAX(missing_months), 0)
    FROM gaps
    """
    loans, missing, max_gap = con.execute(query).fetchone()
    output.write_text(
        "metric,value\n"
        f"loans_with_month_gaps,{loans}\n"
        f"total_missing_months,{missing}\n"
        f"max_gap_months,{max_gap}\n"
    )
    return {
        "loans_with_month_gaps": int(loans),
        "total_missing_months": int(missing),
        "max_gap_months": int(max_gap),
    }


def _origination_without_performance(con: duckdb.DuckDBPyConnection, output: Path) -> None:
    query = """
    WITH o AS (
      SELECT _source_year, loan_id, first_payment_date, maturity_date, origination_month,
             original_upb, seller_name, property_state, loan_purpose
      FROM read_parquet('data/silver/freddie/origination/*/*.parquet')
    ),
    p AS (
      SELECT DISTINCT _source_year, loan_id
      FROM read_parquet('data/silver/freddie/performance/*/*.parquet')
    )
    SELECT o.*
    FROM o ANTI JOIN p USING (_source_year, loan_id)
    ORDER BY _source_year, loan_id
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{output.as_posix()}' (HEADER, DELIMITER ',')")


def _duckdb_summary(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    orig = con.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT loan_id)
        FROM read_parquet('data/silver/freddie/origination/*/*.parquet')
        """
    ).fetchone()
    perf = con.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT loan_id), COUNT(DISTINCT loan_id || '|' || period),
               MIN(period), MAX(period)
        FROM read_parquet('data/silver/freddie/performance/*/*.parquet')
        """
    ).fetchone()
    return {
        "origination_rows": orig[0],
        "origination_unique_loans": orig[1],
        "performance_rows": perf[0],
        "performance_unique_loans": perf[1],
        "performance_unique_loan_periods": perf[2],
        "min_period": str(perf[3]),
        "max_period": str(perf[4]),
    }


def _write_summary(
    manifest_dir: Path,
    elapsed: float,
    silver_root: Path,
    duckdb_summary: dict[str, Any],
    temporal_summary: dict[str, int],
) -> None:
    silver_size = sum(path.stat().st_size for path in silver_root.rglob("*.parquet"))
    lines = [
        "# Silver Build Summary",
        "",
        f"- Origination rows: {duckdb_summary['origination_rows']}",
        f"- Performance rows: {duckdb_summary['performance_rows']}",
        f"- Origination unique loans: {duckdb_summary['origination_unique_loans']}",
        f"- Performance unique loans: {duckdb_summary['performance_unique_loans']}",
        f"- Min performance period: {duckdb_summary['min_period']}",
        f"- Max performance period: {duckdb_summary['max_period']}",
        f"- Loans with performance month gaps: {temporal_summary['loans_with_month_gaps']}",
        f"- Total missing performance months: {temporal_summary['total_missing_months']}",
        f"- Max gap months: {temporal_summary['max_gap_months']}",
        f"- Silver disk size bytes: {silver_size}",
        f"- Processing time seconds: {elapsed:.2f}",
    ]
    (manifest_dir / "silver_summary.md").write_text("\n".join(lines) + "\n")


def run_freddie_silver_build(
    *,
    year: int | None = None,
    all_years: bool = False,
    force: bool = False,
    repo_root: Path | None = None,
) -> list[SilverPartitionResult]:
    """Build Freddie Mac Silver partitions from Bronze partitions."""
    configure_logging()
    root = find_repo_root(repo_root)
    config = _load_config(root)
    bronze_root = root / config["ingestion"]["bronze_freddie_dir"]
    silver_root = root / config["ingestion"]["silver_freddie_dir"]
    manifest_dir = root / config["ingestion"]["silver_manifest_dir"]
    available_years = _discover_years(bronze_root)
    years = available_years if all_years else [year]
    if not years or years == [None]:
        msg = "Specify a year or all_years=True"
        raise SilverBuildError(msg)
    missing = [selected for selected in years if selected not in available_years]
    if missing:
        msg = f"Missing Bronze partitions for years: {missing}"
        raise SilverBuildError(msg)

    started = time.perf_counter()
    results: list[SilverPartitionResult] = []
    dq_rows: list[SilverDQMetrics] = []
    normalization_rows: list[dict[str, Any]] = []
    for selected_year in years:
        year_results, year_dq, year_norm = _process_year(
            selected_year,
            bronze_root,
            silver_root,
            force,
        )
        results.extend(year_results)
        dq_rows.extend(year_dq)
        normalization_rows.extend(year_norm)

    _write_manifest(manifest_dir / "silver_manifest.json", results)
    _write_csv(manifest_dir / "silver_data_quality.csv", [row.as_dict() for row in dq_rows])
    _write_csv(manifest_dir / "normalization_summary.csv", normalization_rows)

    with duckdb.connect(database=":memory:") as con:
        con.execute(f"SET home_directory='{root.as_posix()}'")
        con.execute("PRAGMA database_list")
        con.execute("SET search_path='main'")
        # Use repository-relative paths for the validation queries.
        current = Path.cwd()
        try:
            os.chdir(root)
            _historical_availability(
                con,
                manifest_dir / "historical_field_availability.csv",
            )
            temporal_summary = _temporal_gap_report(
                con,
                manifest_dir / "performance_temporal_gaps.csv",
            )
            _origination_without_performance(
                con,
                manifest_dir / "origination_without_performance.csv",
            )
            duck_summary = _duckdb_summary(con)
        finally:
            os.chdir(current)

    _write_summary(
        manifest_dir,
        time.perf_counter() - started,
        silver_root,
        duck_summary,
        temporal_summary,
    )
    LOGGER.info("Freddie Mac Silver build completed")
    return results
