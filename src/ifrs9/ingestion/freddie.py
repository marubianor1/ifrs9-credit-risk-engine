"""Freddie Mac Source to Bronze ingestion pipeline."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import yaml

from ifrs9.ingestion.schema_registry import DatasetKind, SourceSchema, find_repo_root, load_schema
from ifrs9.ingestion.source_inventory import (
    discover_sample_zips,
    expected_members_for_year,
    parse_sample_year,
)

LOGGER = logging.getLogger(__name__)


class FreddieIngestionError(RuntimeError):
    """Raised when Freddie Mac Bronze ingestion fails."""


@dataclass(frozen=True)
class DatasetQualityMetrics:
    """Data-quality metrics for one ingested dataset/year."""

    year: int
    dataset: DatasetKind
    row_count: int
    column_count: int
    unique_loan_count: int
    duplicate_loan_ids: int | None
    unique_loan_period_count: int | None
    duplicate_loan_periods: int | None
    null_loan_ids: int
    null_periods: int | None
    min_period: str | None
    max_period: str | None
    source_zip_size: int
    parquet_size: int
    processing_time_seconds: float
    performance_loans_without_origination: int | None = None
    origination_loans_without_performance: int | None = None


@dataclass(frozen=True)
class YearIngestionResult:
    """Manifest entry for one vintage year."""

    year: int
    source_zip: str
    schema_version: str
    origination_rows: int
    performance_rows: int
    origination_parquet: str
    performance_parquet: str
    status: str
    processing_time_seconds: float


def configure_logging() -> None:
    """Configure readable structured logging for command-line ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "data.yaml").open() as stream:
        return yaml.safe_load(stream)


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _parquet_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*.parquet"))


def _schema_column_names(schema: SourceSchema) -> list[str]:
    return [column.canonical_name for column in schema.columns]


def _target_dtype(target_type: str) -> pl.DataType:
    normalized = target_type.lower()
    if normalized == "integer":
        return pl.Int64
    if normalized == "decimal":
        return pl.Float64
    return pl.Utf8


def _apply_safe_types(frame: pl.DataFrame, schema: SourceSchema) -> pl.DataFrame:
    expressions = []
    for column in schema.columns:
        dtype = _target_dtype(column.target_type)
        if dtype == pl.Utf8:
            expressions.append(pl.col(column.canonical_name).cast(pl.Utf8))
        else:
            expressions.append(pl.col(column.canonical_name).cast(dtype, strict=False))
    return frame.with_columns(expressions)


def _validate_column_counts(data: bytes, expected_count: int, member: str) -> int:
    row_count = 0
    for row_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line:
            continue
        observed = raw_line.decode("utf-8", errors="replace").count("|") + 1
        if observed != expected_count:
            msg = (
                f"{member} row {row_number} has {observed} columns; "
                f"expected {expected_count}"
            )
            raise FreddieIngestionError(msg)
        row_count += 1
    return row_count


def read_member_to_frame(
    zip_path: Path,
    member: str,
    schema: SourceSchema,
    year: int,
    ingested_at: str,
) -> pl.DataFrame:
    """Read one ZIP member into a schema-mapped Bronze dataframe."""
    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(member)

    _validate_column_counts(data, schema.expected_column_count, member)
    frame = pl.read_csv(
        BytesIO(data),
        separator=schema.delimiter,
        has_header=False,
        new_columns=_schema_column_names(schema),
        infer_schema_length=0,
        empty_string_is_null=False,
    )
    frame = _apply_safe_types(frame, schema)
    return frame.with_columns(
        pl.lit(year).alias("_source_year"),
        pl.lit(zip_path.name).alias("_source_zip"),
        pl.lit(member).alias("_source_member"),
        pl.lit(ingested_at).alias("_ingested_at"),
        pl.lit(schema.version).alias("_schema_version"),
        (pl.int_range(pl.len(), dtype=pl.Int64) + 1).alias("_source_row_number"),
    )


def _write_parquet_atomic(
    frame: pl.DataFrame,
    final_dir: Path,
    dataset: DatasetKind,
    force: bool,
) -> Path:
    final_file = final_dir / f"part-{dataset}.parquet"
    if final_file.exists() and not force:
        return final_file

    tmp_dir = final_dir.parent / f".{final_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    tmp_file = tmp_dir / final_file.name
    try:
        frame.write_parquet(tmp_file, compression="zstd")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return final_file


def _dataset_metrics(
    frame: pl.DataFrame,
    dataset: DatasetKind,
    year: int,
    source_zip_size: int,
    parquet_dir: Path,
    elapsed: float,
) -> DatasetQualityMetrics:
    loan_null_filter = pl.col("loan_id").is_null() | (pl.col("loan_id") == "")
    unique_loan_count = frame.select(pl.col("loan_id").n_unique()).item()
    null_loan_ids = frame.filter(loan_null_filter).height
    duplicate_loan_ids = None
    unique_loan_period_count = None
    duplicate_loan_periods = None
    null_periods = None
    min_period = None
    max_period = None

    if dataset == "origination":
        duplicate_loan_ids = frame.height - unique_loan_count
    else:
        period_null_filter = pl.col("period").is_null() | (pl.col("period") == "")
        null_periods = frame.filter(period_null_filter).height
        unique_loan_period_count = frame.select(pl.struct("loan_id", "period").n_unique()).item()
        duplicate_loan_periods = frame.height - unique_loan_period_count
        min_period = frame.select(pl.col("period").min()).item()
        max_period = frame.select(pl.col("period").max()).item()

    return DatasetQualityMetrics(
        year=year,
        dataset=dataset,
        row_count=frame.height,
        column_count=len([name for name in frame.columns if not name.startswith("_")]),
        unique_loan_count=unique_loan_count,
        duplicate_loan_ids=duplicate_loan_ids,
        unique_loan_period_count=unique_loan_period_count,
        duplicate_loan_periods=duplicate_loan_periods,
        null_loan_ids=null_loan_ids,
        null_periods=null_periods,
        min_period=min_period,
        max_period=max_period,
        source_zip_size=source_zip_size,
        parquet_size=_parquet_size(parquet_dir),
        processing_time_seconds=elapsed,
    )


def _referential_metrics(
    origination: pl.DataFrame,
    performance: pl.DataFrame,
) -> tuple[int, int]:
    orig_loans = origination.select("loan_id").unique()
    perf_loans = performance.select("loan_id").unique()
    performance_unmatched = perf_loans.join(orig_loans, on="loan_id", how="anti").height
    origination_unmatched = orig_loans.join(perf_loans, on="loan_id", how="anti").height
    return performance_unmatched, origination_unmatched


def validate_parquet_with_duckdb(parquet_path: Path, dataset: DatasetKind) -> dict[str, Any]:
    """Run lightweight DuckDB validation over a Bronze Parquet file."""
    query_path = parquet_path.as_posix()
    with duckdb.connect(database=":memory:") as con:
        if dataset == "origination":
            row = con.execute(
                "SELECT COUNT(*) AS row_count, COUNT(DISTINCT loan_id) AS unique_loans "
                "FROM read_parquet(?)",
                [query_path],
            ).fetchone()
            return {"row_count": row[0], "unique_loans": row[1]}
        row = con.execute(
            "SELECT COUNT(*) AS row_count, COUNT(DISTINCT loan_id) AS unique_loans, "
            "MIN(period) AS min_period, MAX(period) AS max_period FROM read_parquet(?)",
            [query_path],
        ).fetchone()
        return {
            "row_count": row[0],
            "unique_loans": row[1],
            "min_period": row[2],
            "max_period": row[3],
        }


def _write_manifest(manifest_dir: Path, results: list[YearIngestionResult]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [asdict(result) for result in results],
    }
    (manifest_dir / "bronze_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_dq_report(manifest_dir: Path, metrics: list[DatasetQualityMetrics]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "bronze_data_quality.csv"
    fieldnames = list(asdict(metrics[0]).keys()) if metrics else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))


def _write_summary(
    manifest_dir: Path,
    metrics: list[DatasetQualityMetrics],
    total_processing_seconds: float,
    source_disk_size: int,
    bronze_disk_size: int,
) -> None:
    orig = [metric for metric in metrics if metric.dataset == "origination"]
    perf = [metric for metric in metrics if metric.dataset == "performance"]
    total_orig_rows = sum(metric.row_count for metric in orig)
    total_perf_rows = sum(metric.row_count for metric in perf)
    total_duplicate_orig = sum(metric.duplicate_loan_ids or 0 for metric in orig)
    total_duplicate_perf = sum(metric.duplicate_loan_periods or 0 for metric in perf)
    total_unmatched_perf = sum(metric.performance_loans_without_origination or 0 for metric in perf)
    min_periods = [metric.min_period for metric in perf if metric.min_period]
    max_periods = [metric.max_period for metric in perf if metric.max_period]

    lines = [
        "# Bronze Ingestion Summary",
        "",
        f"- Years processed: {', '.join(str(metric.year) for metric in orig)}",
        f"- Origination total rows: {total_orig_rows}",
        f"- Performance total rows: {total_perf_rows}",
        f"- Unique loans: {sum(metric.unique_loan_count for metric in orig)}",
        f"- Loan-month observations: {total_perf_rows}",
        f"- Duplicate origination loan IDs: {total_duplicate_orig}",
        f"- Duplicate performance `(loan_id, period)` records: {total_duplicate_perf}",
        f"- Unmatched performance loan IDs: {total_unmatched_perf}",
        f"- Min reporting month: {min(min_periods) if min_periods else ''}",
        f"- Max reporting month: {max(max_periods) if max_periods else ''}",
        f"- Source disk size bytes: {source_disk_size}",
        f"- Bronze disk size bytes: {bronze_disk_size}",
        f"- Processing time seconds: {total_processing_seconds:.2f}",
        (
            f"- Compression ratio: {source_disk_size / bronze_disk_size:.2f}"
            if bronze_disk_size
            else "- Compression ratio: n/a"
        ),
        "",
        "| Year | Origination rows | Performance rows | Duplicate orig IDs | "
        "Duplicate loan-periods | Unmatched perf loans | Min period | Max period |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    by_year = {metric.year: {} for metric in metrics}
    for metric in metrics:
        by_year[metric.year][metric.dataset] = metric
    for year in sorted(by_year):
        orig_metric = by_year[year]["origination"]
        perf_metric = by_year[year]["performance"]
        lines.append(
            f"| {year} | {orig_metric.row_count} | {perf_metric.row_count} | "
            f"{orig_metric.duplicate_loan_ids or 0} | {perf_metric.duplicate_loan_periods or 0} | "
            f"{perf_metric.performance_loans_without_origination or 0} | "
            f"{perf_metric.min_period or ''} | {perf_metric.max_period or ''} |"
        )
    (manifest_dir / "bronze_summary.md").write_text("\n".join(lines) + "\n")


def _process_year(
    year: int,
    zip_path: Path,
    repo_root: Path,
    bronze_root: Path,
    force: bool,
) -> tuple[YearIngestionResult, list[DatasetQualityMetrics]]:
    LOGGER.info("Processing vintage %s", year)
    start = time.perf_counter()
    orig_member, perf_member = expected_members_for_year(year)
    orig_schema = load_schema("origination", year=year, repo_root=repo_root)
    perf_schema = load_schema("performance", year=year, repo_root=repo_root)
    ingested_at = datetime.now(UTC).isoformat()
    source_zip_size = zip_path.stat().st_size

    orig_dir = bronze_root / "origination" / f"vintage_year={year}"
    perf_dir = bronze_root / "performance" / f"vintage_year={year}"
    orig_file = orig_dir / "part-origination.parquet"
    perf_file = perf_dir / "part-performance.parquet"
    if orig_file.exists() and perf_file.exists() and not force:
        LOGGER.info("Vintage %s already has Bronze outputs; skipping without --force", year)
        duck_orig = validate_parquet_with_duckdb(orig_file, "origination")
        duck_perf = validate_parquet_with_duckdb(perf_file, "performance")
        elapsed = time.perf_counter() - start
        result = YearIngestionResult(
            year=year,
            source_zip=_relative(zip_path, repo_root),
            schema_version=orig_schema.version,
            origination_rows=duck_orig["row_count"],
            performance_rows=duck_perf["row_count"],
            origination_parquet=_relative(orig_file, repo_root),
            performance_parquet=_relative(perf_file, repo_root),
            status="skipped_existing",
            processing_time_seconds=elapsed,
        )
        return result, []

    LOGGER.info("Reading vintage %s origination", year)
    orig_frame = read_member_to_frame(zip_path, orig_member, orig_schema, year, ingested_at)
    LOGGER.info("Reading vintage %s performance", year)
    perf_frame = read_member_to_frame(zip_path, perf_member, perf_schema, year, ingested_at)

    LOGGER.info("Writing vintage %s Bronze Parquet", year)
    _write_parquet_atomic(orig_frame, orig_dir, "origination", force=force)
    _write_parquet_atomic(perf_frame, perf_dir, "performance", force=force)
    duck_orig = validate_parquet_with_duckdb(orig_file, "origination")
    duck_perf = validate_parquet_with_duckdb(perf_file, "performance")
    if duck_orig["row_count"] != orig_frame.height or duck_perf["row_count"] != perf_frame.height:
        msg = f"DuckDB row-count validation failed for vintage {year}"
        raise FreddieIngestionError(msg)

    elapsed = time.perf_counter() - start
    orig_metric = _dataset_metrics(
        orig_frame, "origination", year, source_zip_size, orig_dir, elapsed
    )
    perf_metric = _dataset_metrics(
        perf_frame, "performance", year, source_zip_size, perf_dir, elapsed
    )
    unmatched_perf, unmatched_orig = _referential_metrics(orig_frame, perf_frame)
    perf_metric = DatasetQualityMetrics(
        **{
            **asdict(perf_metric),
            "performance_loans_without_origination": unmatched_perf,
            "origination_loans_without_performance": unmatched_orig,
        }
    )

    LOGGER.info("Vintage %s completed", year)
    result = YearIngestionResult(
        year=year,
        source_zip=_relative(zip_path, repo_root),
        schema_version=orig_schema.version,
        origination_rows=orig_frame.height,
        performance_rows=perf_frame.height,
        origination_parquet=_relative(orig_file, repo_root),
        performance_parquet=_relative(perf_file, repo_root),
        status="success",
        processing_time_seconds=elapsed,
    )
    return result, [orig_metric, perf_metric]


def run_freddie_ingestion(
    *,
    year: int | None = None,
    all_years: bool = False,
    force: bool = False,
    repo_root: Path | None = None,
) -> list[YearIngestionResult]:
    """Run Freddie Mac Source to Bronze ingestion."""
    configure_logging()
    root = find_repo_root(repo_root)
    config = _load_config(root)
    source_dir = root / config["source_data"]["freddie_mac"]["source_dir"]
    bronze_root = root / config["ingestion"]["bronze_freddie_dir"]
    manifest_dir = root / config["ingestion"]["manifest_dir"]

    LOGGER.info("Discovering Freddie Mac source files")
    zip_paths = {parse_sample_year(path): path for path in discover_sample_zips(source_dir)}
    LOGGER.info("Found %s annual sample ZIPs", len(zip_paths))
    years = sorted(zip_paths) if all_years else [year]
    if not years or years == [None]:
        msg = "Specify a year or all_years=True"
        raise FreddieIngestionError(msg)
    missing = [selected for selected in years if selected not in zip_paths]
    if missing:
        msg = f"Missing source ZIPs for years: {missing}"
        raise FreddieIngestionError(msg)

    total_start = time.perf_counter()
    results: list[YearIngestionResult] = []
    metrics: list[DatasetQualityMetrics] = []
    for selected_year in years:
        result, year_metrics = _process_year(
            selected_year,
            zip_paths[selected_year],
            root,
            bronze_root,
            force,
        )
        results.append(result)
        metrics.extend(year_metrics)

    _write_manifest(manifest_dir, results)
    if metrics:
        _write_dq_report(manifest_dir, metrics)
        source_disk_size = sum(zip_paths[selected_year].stat().st_size for selected_year in years)
        bronze_disk_size = _parquet_size(bronze_root)
        _write_summary(
            manifest_dir,
            metrics,
            time.perf_counter() - total_start,
            source_disk_size,
            bronze_disk_size,
        )
    LOGGER.info("Freddie Mac Bronze ingestion completed")
    return results
