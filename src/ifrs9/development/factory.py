"""Temporal development sample and split factory."""

from __future__ import annotations

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

from ifrs9.development.config import DevelopmentConfig, load_development_config
from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.feature_registry import load_default_registry

LOGGER = logging.getLogger(__name__)


class DevelopmentBuildError(RuntimeError):
    """Raised when development sample construction fails."""


@dataclass(frozen=True)
class DevelopmentBuildResult:
    """Result metadata for a development sample build."""

    rows: int
    started_at: str
    completed_at: str
    status: str
    processing_time_seconds: float


def configure_logging() -> None:
    """Configure command-line logging for development sample builds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_data_config(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "data.yaml").open() as stream:
        return yaml.safe_load(stream)


def _prepare_tmp(final_dir: Path) -> Path:
    tmp_dir = final_dir.parent / f".{final_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


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


def _git_state(repo_root: Path) -> dict[str, str]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}
    return {
        "commit": commit_result.stdout.strip(),
        "dirty": str(bool(status_result.stdout.strip())),
    }


def _snapshot_filter(config: DevelopmentConfig) -> str:
    if config.snapshot_frequency == "monthly":
        return "true"
    if config.snapshot_frequency == "quarter_end":
        return "month(as_of_date) IN (3, 6, 9, 12)"
    if config.snapshot_frequency == "year_end":
        return "month(as_of_date) = 12"
    msg = f"Unsupported snapshot frequency: {config.snapshot_frequency}"
    raise DevelopmentBuildError(msg)


def _split_case(config: DevelopmentConfig) -> str:
    windows = config.split_windows
    return f"""
        CASE
            WHEN as_of_date BETWEEN DATE '{windows.train.start}' AND DATE '{windows.train.end}'
                THEN 'TRAIN'
            WHEN as_of_date BETWEEN DATE '{windows.validation.start}'
                 AND DATE '{windows.validation.end}'
                THEN 'VALIDATION'
            WHEN as_of_date BETWEEN DATE '{windows.oot.start}' AND DATE '{windows.oot.end}'
                THEN 'OOT'
            ELSE 'EXCLUDED'
        END
    """


def _sampling_probability_sql(config: DevelopmentConfig) -> str:
    if config.sampling.strategy == "none":
        return "1.0"
    fraction = config.sampling.nondefault_sample_fraction
    if config.sampling.strategy == "random_nondefault":
        return f"CASE WHEN default_next_12m = 1 THEN 1.0 ELSE {fraction} END"
    if config.sampling.strategy == "stratified_nondefault":
        return f"""
            CASE
                WHEN default_next_12m = 1 THEN 1.0
                WHEN split = 'EXCLUDED' THEN 1.0
                WHEN nondefault_stratum_count <= {config.sampling.minimum_nondefaults_per_stratum}
                    THEN 1.0
                ELSE {fraction}
            END
        """
    msg = f"Unsupported sampling strategy: {config.sampling.strategy}"
    raise DevelopmentBuildError(msg)


def _post_population_ctes(config: DevelopmentConfig) -> str:
    if config.loan_disjoint:
        split_cte = f"""
    disjoint_seed AS (
        SELECT
            *,
            min(CASE WHEN raw_split = 'TRAIN' THEN 1
                     WHEN raw_split = 'VALIDATION' THEN 2
                     WHEN raw_split = 'OOT' THEN 3
                     ELSE 4 END) OVER (PARTITION BY loan_id) AS first_split_rank
        FROM population
    ),
    split_assigned AS (
        SELECT
            *,
            CASE
                WHEN {str(config.loan_disjoint).lower()} AND first_split_rank = 1
                     AND raw_split <> 'TRAIN' THEN 'EXCLUDED'
                WHEN {str(config.loan_disjoint).lower()} AND first_split_rank = 2
                     AND raw_split NOT IN ('VALIDATION', 'EXCLUDED') THEN 'EXCLUDED'
                WHEN {str(config.loan_disjoint).lower()} AND first_split_rank = 3
                     AND raw_split NOT IN ('OOT', 'EXCLUDED') THEN 'EXCLUDED'
                ELSE raw_split
            END AS split
        FROM disjoint_seed
    ),
"""
    else:
        split_cte = """
    split_assigned AS (
        SELECT *, raw_split AS split
        FROM population
    ),
"""
    if config.sampling.strategy == "none":
        sampling_cte = """
    sampled AS (
        SELECT
            *,
            1.0 AS deterministic_random,
            1.0 AS sampling_probability
        FROM split_assigned
    )
"""
    elif config.sampling.strategy == "random_nondefault":
        sampling_probability = _sampling_probability_sql(config)
        sampling_cte = f"""
    sampled AS (
        SELECT
            *,
            hash(loan_id || '|' || CAST(as_of_date AS VARCHAR) || '|{config.sampling.seed}')
                / 18446744073709551615.0 AS deterministic_random,
            {sampling_probability} AS sampling_probability
        FROM split_assigned
    )
"""
    else:
        sampling_probability = _sampling_probability_sql(config)
        sampling_cte = f"""
    stratum_counts AS (
        SELECT
            *,
            count(*) FILTER (WHERE default_next_12m = 0) OVER (
                PARTITION BY split, year(as_of_date)
            ) AS nondefault_stratum_count,
            hash(loan_id || '|' || CAST(as_of_date AS VARCHAR) || '|{config.sampling.seed}')
                / 18446744073709551615.0 AS deterministic_random
        FROM split_assigned
    ),
    sampled AS (
        SELECT
            *,
            {sampling_probability} AS sampling_probability
        FROM stratum_counts
    )
"""
    return split_cte + sampling_cte


def _development_sql(config: DevelopmentConfig) -> str:
    split_case = _split_case(config)
    snapshot_filter = _snapshot_filter(config)
    full_horizon = config.binary_12m_full_horizon_cutoff.isoformat()
    source_start = min(
        config.split_windows.train.start,
        config.split_windows.validation.start,
        config.split_windows.oot.start,
    )
    source_end = max(
        config.split_windows.train.end,
        config.split_windows.validation.end,
        config.split_windows.oot.end,
    )
    processed_at = datetime.now(UTC).isoformat()
    post_population_ctes = _post_population_ctes(config)
    if config.population == "application":
        population_cte = """
    ranked_population AS (
        SELECT
            *,
            row_number() OVER (PARTITION BY loan_id ORDER BY as_of_date) AS population_rank
        FROM eligible
    ),
    population AS (
        SELECT *
        FROM ranked_population
        WHERE population_rank = 1
    ),
"""
    else:
        population_cte = """
    population AS (
        SELECT *
        FROM eligible
    ),
"""
    return f"""
    WITH source AS (
        SELECT
            loan_id,
            as_of_date,
            eligible_for_12m_pd,
            target_12m_observable,
            default_next_12m,
            event_censored,
            event_type
        FROM pd_12m_targets
        WHERE as_of_date BETWEEN DATE '{source_start}' AND DATE '{source_end}'
          AND {snapshot_filter}
    ),
    eligible AS (
        SELECT
            *,
            eligible_for_12m_pd
            AND target_12m_observable
            AND default_next_12m IS NOT NULL
            AND as_of_date <= DATE '{full_horizon}'
                AS development_eligible_12m,
            {split_case} AS raw_split
        FROM source
    ),
{population_cte}
{post_population_ctes}
    SELECT
        loan_id,
        as_of_date,
        split,
        '{config.snapshot_frequency}' AS snapshot_frequency,
        '{config.population}' AS population,
        development_eligible_12m,
        default_next_12m,
        CASE
            WHEN NOT development_eligible_12m THEN false
            WHEN default_next_12m = 1 THEN true
            WHEN split = 'EXCLUDED' THEN false
            WHEN deterministic_random <= sampling_probability THEN true
            ELSE false
        END AS sample_selected,
        CASE WHEN development_eligible_12m THEN sampling_probability ELSE NULL END
            AS sampling_probability,
        CASE WHEN development_eligible_12m THEN 1.0 / sampling_probability ELSE NULL END
            AS sample_weight,
        target_12m_observable,
        event_censored,
        event_type,
        year(as_of_date) AS observation_year,
        '{processed_at}' AS _development_processed_at
    FROM sampled
    """


def _register_target_view(con: duckdb.DuckDBPyConnection, target_root: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW pd_12m_targets AS
        SELECT * FROM read_parquet('{target_root.as_posix()}/pd_12m_targets/*.parquet')
        """
    )


def _register_development_view(con: duckdb.DuckDBPyConnection, development_dir: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW development_sample AS
        SELECT * FROM read_parquet('{development_dir.as_posix()}/*.parquet')
        """
    )


def _write_split_profile(con: duckdb.DuckDBPyConnection, artifact_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT
              split,
              observation_year,
              COUNT(*) AS rows,
              COUNT(DISTINCT loan_id) AS loans,
              SUM(CASE WHEN development_eligible_12m THEN 1 ELSE 0 END) AS eligible_rows,
              SUM(CASE WHEN development_eligible_12m AND default_next_12m = 1
                       THEN 1 ELSE 0 END) AS defaults,
              SUM(CASE WHEN development_eligible_12m AND default_next_12m = 0
                       THEN 1 ELSE 0 END) AS nondefaults,
              SUM(CASE WHEN development_eligible_12m AND default_next_12m = 1
                       THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN development_eligible_12m THEN 1 ELSE 0 END), 0)
                AS bad_rate,
              SUM(CASE WHEN NOT target_12m_observable THEN 1 ELSE 0 END) AS censored_rows,
              SUM(CASE WHEN NOT development_eligible_12m THEN 1 ELSE 0 END) AS excluded_rows,
              SUM(CASE WHEN development_eligible_12m AND default_next_12m = 0
                       THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN development_eligible_12m AND default_next_12m = 1
                                  THEN 1 ELSE 0 END), 0)
                AS nondefault_to_default_ratio
          FROM development_sample
          GROUP BY 1, 2
          ORDER BY 1, 2
        )
        TO '{(artifact_dir / "split_profile.csv").as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )


def _write_sample_profile(con: duckdb.DuckDBPyConnection, artifact_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          WITH split_sets AS (
            SELECT loan_id,
                   max(split = 'TRAIN') AS in_train,
                   max(split = 'VALIDATION') AS in_validation,
                   max(split = 'OOT') AS in_oot
            FROM development_sample
            WHERE development_eligible_12m
            GROUP BY 1
          ),
          metrics AS (
            SELECT 'rows' AS metric, split, COUNT(*)::DOUBLE AS value
            FROM development_sample GROUP BY 2
            UNION ALL
            SELECT 'loans', split, COUNT(DISTINCT loan_id)::DOUBLE
            FROM development_sample GROUP BY 2
            UNION ALL
            SELECT 'selected_rows', split, SUM(CASE WHEN sample_selected THEN 1 ELSE 0 END)::DOUBLE
            FROM development_sample GROUP BY 2
            UNION ALL
            SELECT 'defaults', split,
                   SUM(CASE WHEN development_eligible_12m AND default_next_12m = 1
                            THEN 1 ELSE 0 END)::DOUBLE
            FROM development_sample GROUP BY 2
            UNION ALL
            SELECT 'nondefaults', split,
                   SUM(CASE WHEN development_eligible_12m AND default_next_12m = 0
                            THEN 1 ELSE 0 END)::DOUBLE
            FROM development_sample GROUP BY 2
            UNION ALL
            SELECT 'loans_in_multiple_splits', 'ALL',
                   SUM(CASE WHEN (in_train::INT + in_validation::INT + in_oot::INT) > 1
                            THEN 1 ELSE 0 END)::DOUBLE
            FROM split_sets
            UNION ALL
            SELECT 'train_validation_overlap', 'ALL',
                   SUM(CASE WHEN in_train AND in_validation THEN 1 ELSE 0 END)::DOUBLE
            FROM split_sets
            UNION ALL
            SELECT 'train_oot_overlap', 'ALL',
                   SUM(CASE WHEN in_train AND in_oot THEN 1 ELSE 0 END)::DOUBLE
            FROM split_sets
            UNION ALL
            SELECT 'validation_oot_overlap', 'ALL',
                   SUM(CASE WHEN in_validation AND in_oot THEN 1 ELSE 0 END)::DOUBLE
            FROM split_sets
          )
          SELECT * FROM metrics
          ORDER BY metric, split
        )
        TO '{(artifact_dir / "sample_profile.csv").as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )


def _collect_validation(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    total = con.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT loan_id), MAX(as_of_date)
        FROM development_sample
        WHERE development_eligible_12m
        """
    ).fetchone()
    by_split = con.execute(
        """
        SELECT split, COUNT(*), COUNT(DISTINCT loan_id),
               SUM(CASE WHEN default_next_12m = 1 THEN 1 ELSE 0 END),
               AVG(default_next_12m)
        FROM development_sample
        WHERE development_eligible_12m
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    overlaps = con.execute(
        """
        WITH split_sets AS (
            SELECT loan_id,
                   max(split = 'TRAIN') AS in_train,
                   max(split = 'VALIDATION') AS in_validation,
                   max(split = 'OOT') AS in_oot
            FROM development_sample
            WHERE development_eligible_12m
            GROUP BY 1
        )
        SELECT
            SUM(CASE WHEN (in_train::INT + in_validation::INT + in_oot::INT) > 1
                     THEN 1 ELSE 0 END),
            SUM(CASE WHEN in_train AND in_validation THEN 1 ELSE 0 END),
            SUM(CASE WHEN in_train AND in_oot THEN 1 ELSE 0 END),
            SUM(CASE WHEN in_validation AND in_oot THEN 1 ELSE 0 END)
        FROM split_sets
        """
    ).fetchone()
    return {
        "eligible_rows": total[0],
        "eligible_loans": total[1],
        "max_eligible_as_of_date": str(total[2]),
        "by_split": [
            {
                "split": row[0],
                "rows": row[1],
                "loans": row[2],
                "defaults": row[3],
                "bad_rate": row[4],
            }
            for row in by_split
        ],
        "loans_in_multiple_splits": overlaps[0],
        "train_validation_overlap": overlaps[1],
        "train_oot_overlap": overlaps[2],
        "validation_oot_overlap": overlaps[3],
    }


def _write_manifest(
    artifact_dir: Path,
    result: DevelopmentBuildResult,
    config: DevelopmentConfig,
    git_state: dict[str, str],
    approved_features: list[str],
    validation: dict[str, object],
) -> None:
    payload = {
        "status": "success",
        "build_timestamp": datetime.now(UTC).isoformat(),
        "config_version": config.version,
        "population": config.population,
        "snapshot_frequency": config.snapshot_frequency,
        "sampling_strategy": config.sampling.strategy,
        "loan_disjoint": config.loan_disjoint,
        "full_horizon_cutoff": config.binary_12m_full_horizon_cutoff.isoformat(),
        "approved_features": approved_features,
        "git": git_state,
        "result": asdict(result),
        "validation": validation,
    }
    (artifact_dir / "development_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _apply_overrides(
    config: DevelopmentConfig,
    *,
    snapshot_frequency: str | None,
    sampling_strategy: str | None,
    population: str | None,
    loan_disjoint: bool | None,
) -> DevelopmentConfig:
    updates: dict[str, object] = {}
    if snapshot_frequency:
        updates["snapshot_frequency"] = snapshot_frequency
    if population:
        updates["population"] = population
    if loan_disjoint is not None:
        updates["loan_disjoint"] = loan_disjoint
    if sampling_strategy:
        updates["sampling"] = config.sampling.model_copy(update={"strategy": sampling_strategy})
    return config.model_copy(update=updates)


def run_development_sample_build(
    *,
    force: bool = False,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    snapshot_frequency: str | None = None,
    sampling_strategy: str | None = None,
    population: str | None = None,
    loan_disjoint: bool | None = None,
) -> DevelopmentBuildResult:
    """Build temporal development sample metadata."""
    configure_logging()
    root = find_repo_root(repo_root)
    config = _apply_overrides(
        load_development_config(root, config_path),
        snapshot_frequency=snapshot_frequency,
        sampling_strategy=sampling_strategy,
        population=population,
        loan_disjoint=loan_disjoint,
    )
    registry = load_default_registry(root)
    approved_features = registry.get_features_for(config.model_use)
    if not approved_features:
        msg = f"No approved features found for model use: {config.model_use}"
        raise DevelopmentBuildError(msg)

    data_config = _load_data_config(root)
    gold_root = root / data_config["ingestion"]["gold_freddie_dir"]
    target_root = gold_root / "targets"
    development_dir = gold_root / "development"
    artifact_dir = root / "artifacts" / "development"
    if development_dir.exists() and not force:
        msg = "Development sample output already exists; rerun with --force to overwrite"
        raise DevelopmentBuildError(msg)

    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    tmp_dir = _prepare_tmp(development_dir)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Building temporal development sample metadata")
    try:
        with duckdb.connect(database=":memory:") as con:
            con.execute("PRAGMA threads=4")
            _register_target_view(con, target_root)
            _copy_query(con, _development_sql(config), tmp_dir / "part-development.parquet")
            _promote_tmp(tmp_dir, development_dir)
            _register_development_view(con, development_dir)
            _write_split_profile(con, artifact_dir)
            _write_sample_profile(con, artifact_dir)
            validation = _collect_validation(con)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    result = DevelopmentBuildResult(
        rows=validation["eligible_rows"],
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        status="success",
        processing_time_seconds=time.perf_counter() - start,
    )
    _write_manifest(
        artifact_dir,
        result,
        config,
        _git_state(root),
        approved_features,
        validation,
    )
    LOGGER.info("Temporal development sample build completed")
    return result
