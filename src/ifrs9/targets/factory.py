"""Default definition and PD target factory."""

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

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.views import register_gold_views
from ifrs9.targets.default_config import DefaultDefinitionConfig, load_default_definition

LOGGER = logging.getLogger(__name__)


class TargetBuildError(RuntimeError):
    """Raised when target construction fails."""


@dataclass(frozen=True)
class TargetBuildResult:
    """Manifest result for a target build."""

    pd_12m_rows: int
    default_event_rows: int
    started_at: str
    completed_at: str
    status: str
    processing_time_seconds: float


def configure_logging() -> None:
    """Configure command-line logging for target builds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_data_config(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "data.yaml").open() as stream:
        return yaml.safe_load(stream)


def _copy_query(con: duckdb.DuckDBPyConnection, query: str, output_path: Path) -> None:
    con.execute(
        f"""
        COPY ({query})
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


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


def _sql_map_case(column: str, mapping: dict[str, str], default: str = "NULL") -> str:
    branches = " ".join(
        f"WHEN {column} = '{code}' THEN '{label}'" for code, label in mapping.items()
    )
    return f"CASE {branches} ELSE {default} END"


def default_row_state_sql(config: DefaultDefinitionConfig) -> str:
    """Return SQL that classifies observable default and exit states by loan-month."""
    default_reason_case = f"""
        CASE
            WHEN delinquency_status_raw IN ({config.delinquency_status_codes_sql})
                THEN 'REO_ACQUISITION'
            WHEN zero_balance_code = '09' THEN 'REO_DISPOSITION'
            WHEN zero_balance_code = '03' THEN 'SHORT_SALE_OR_CHARGE_OFF'
            WHEN zero_balance_code = '02' THEN 'THIRD_PARTY_SALE'
            WHEN delinquency_months >= {config.delinquency_threshold_months}
                THEN 'DELINQUENCY_{config.delinquency_threshold_months}_PLUS'
            ELSE NULL
        END
    """
    exit_reason_case = _sql_map_case(
        "zero_balance_code",
        config.zero_balance_non_default_exit_codes,
    )
    return f"""
    WITH normalized AS (
        SELECT
            loan_id,
            vintage_year,
            as_of_date,
            delinquency_months,
            nullif(delinquency_status_raw, '') AS delinquency_status_raw,
            nullif(zero_balance_code, '') AS zero_balance_code,
            zero_balance_effective_date,
            has_reporting_gap_to_date
        FROM gold_loan_month
    )
    SELECT
        loan_id,
        vintage_year,
        as_of_date,
        delinquency_months,
        delinquency_status_raw,
        zero_balance_code,
        zero_balance_effective_date,
        has_reporting_gap_to_date,
        coalesce(delinquency_months >= {config.delinquency_threshold_months}, false)
            AS default_due_to_delinquency,
        (
            coalesce(delinquency_status_raw IN ({config.delinquency_status_codes_sql}), false)
            OR coalesce(zero_balance_code IN ({config.default_zero_balance_codes_sql}), false)
        ) AS default_due_to_credit_event,
        (
            coalesce(delinquency_months >= {config.delinquency_threshold_months}, false)
            OR coalesce(delinquency_status_raw IN ({config.delinquency_status_codes_sql}), false)
            OR coalesce(zero_balance_code IN ({config.default_zero_balance_codes_sql}), false)
        ) AS default_flag,
        {default_reason_case} AS default_reason,
        coalesce(zero_balance_code IN ({config.non_default_zero_balance_codes_sql}), false)
            AS non_default_exit_flag,
        {exit_reason_case} AS non_default_exit_reason,
        zero_balance_code IS NOT NULL AS terminal_exit_flag
    FROM normalized
    """


def _target_sql(config: DefaultDefinitionConfig) -> str:
    cutoff = config.reporting_cutoff.isoformat()
    horizon = config.target_horizon_months
    return f"""
    WITH row_state AS (
        {default_row_state_sql(config)}
    ),
    enriched AS (
        SELECT
            *,
            max(default_flag::INTEGER) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) > 0 AS ever_default_to_date,
            min(CASE WHEN default_flag THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id
            ) AS first_default_date,
            max(terminal_exit_flag::INTEGER) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) > 0 AS terminal_exit_to_date,
            min(CASE WHEN default_flag THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ) AS future_first_default_date,
            min(CASE WHEN non_default_exit_flag THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ) AS future_first_non_default_exit_date
        FROM row_state
    ),
    with_future_exit AS (
        SELECT
            e.*,
            x.non_default_exit_reason AS future_first_non_default_exit_reason
        FROM enriched e
        LEFT JOIN row_state x
          ON e.loan_id = x.loan_id
         AND e.future_first_non_default_exit_date = x.as_of_date
         AND x.non_default_exit_flag
    ),
    targets AS (
        SELECT
            *,
            as_of_date + INTERVAL {horizon} MONTH AS target_12m_horizon_date,
            NOT ever_default_to_date AND NOT terminal_exit_to_date AS eligible_for_12m_pd,
            NOT ever_default_to_date AND NOT terminal_exit_to_date AS eligible_for_lifetime_pd,
            CASE
                WHEN future_first_default_date IS NULL THEN NULL
                ELSE date_diff('month', as_of_date, future_first_default_date)
            END AS months_to_default_from_observation,
            CASE
                WHEN future_first_default_date IS NOT NULL
                 AND (
                    future_first_non_default_exit_date IS NULL
                    OR future_first_default_date <= future_first_non_default_exit_date
                 )
                    THEN future_first_default_date
                WHEN future_first_non_default_exit_date IS NOT NULL
                    THEN future_first_non_default_exit_date
                ELSE NULL
            END AS first_future_event_date,
            CASE
                WHEN future_first_default_date IS NOT NULL
                 AND (
                    future_first_non_default_exit_date IS NULL
                    OR future_first_default_date <= future_first_non_default_exit_date
                 )
                    THEN 'DEFAULT'
                WHEN future_first_non_default_exit_date IS NOT NULL
                    THEN future_first_non_default_exit_reason
                ELSE 'CENSORED'
            END AS event_type
        FROM with_future_exit
    )
    SELECT
        loan_id,
        vintage_year,
        as_of_date,
        default_due_to_delinquency,
        default_due_to_credit_event,
        default_flag,
        default_reason,
        first_default_date,
        CASE
            WHEN first_default_date IS NULL THEN NULL
            ELSE date_diff('month', as_of_date, first_default_date)
        END AS months_to_first_default,
        first_default_date IS NOT NULL AS ever_default,
        eligible_for_12m_pd,
        eligible_for_lifetime_pd,
        CASE
            WHEN NOT eligible_for_12m_pd THEN NULL
            WHEN future_first_default_date IS NOT NULL
             AND future_first_default_date <= target_12m_horizon_date
             AND (
                future_first_non_default_exit_date IS NULL
                OR future_first_default_date <= future_first_non_default_exit_date
             )
                THEN 1
            WHEN target_12m_horizon_date <= DATE '{cutoff}'
                THEN 0
            WHEN future_first_non_default_exit_date IS NOT NULL
             AND future_first_non_default_exit_date <= target_12m_horizon_date
                THEN 0
            ELSE NULL
        END AS default_next_12m,
        months_to_default_from_observation,
        CASE
            WHEN NOT eligible_for_12m_pd THEN false
            WHEN future_first_default_date IS NOT NULL
             AND future_first_default_date <= target_12m_horizon_date
             AND (
                future_first_non_default_exit_date IS NULL
                OR future_first_default_date <= future_first_non_default_exit_date
             )
                THEN true
            WHEN target_12m_horizon_date <= DATE '{cutoff}'
                THEN true
            WHEN future_first_non_default_exit_date IS NOT NULL
             AND future_first_non_default_exit_date <= target_12m_horizon_date
                THEN true
            ELSE false
        END AS target_12m_observable,
        event_type = 'DEFAULT' AS event_default,
        event_type NOT IN ('DEFAULT', 'CENSORED') AS event_prepayment_or_exit,
        event_type = 'CENSORED' AS event_censored,
        CASE
            WHEN first_future_event_date IS NOT NULL
                THEN date_diff('month', as_of_date, first_future_event_date)
            ELSE date_diff('month', as_of_date, DATE '{cutoff}')
        END AS months_until_event,
        event_type,
        future_first_non_default_exit_date,
        future_first_non_default_exit_reason,
        has_reporting_gap_to_date,
        '{datetime.now(UTC).isoformat()}' AS _target_processed_at
    FROM targets
    """


def _default_events_sql(config: DefaultDefinitionConfig) -> str:
    probation = config.cure.probation_months
    return f"""
    WITH row_state AS (
        {default_row_state_sql(config)}
    ),
    ordered AS (
        SELECT
            *,
            lag(as_of_date) OVER (PARTITION BY loan_id ORDER BY as_of_date) AS prior_as_of_date
        FROM row_state
    ),
    streak_breaks AS (
        SELECT
            *,
            CASE
                WHEN default_flag THEN 1
                WHEN prior_as_of_date IS NULL THEN 1
                WHEN date_diff('month', prior_as_of_date, as_of_date) <> 1 THEN 1
                ELSE 0
            END AS non_default_streak_break
        FROM ordered
    ),
    streak_groups AS (
        SELECT
            *,
            sum(non_default_streak_break) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS non_default_streak_group
        FROM streak_breaks
    ),
    streaks AS (
        SELECT
            *,
            CASE
                WHEN default_flag THEN 0
                ELSE row_number() OVER (
                    PARTITION BY loan_id, non_default_streak_group ORDER BY as_of_date
                )
            END AS consecutive_non_default_months
        FROM streak_groups
    ),
    rows_with_cure_candidate AS (
        SELECT
            *,
            min(CASE WHEN consecutive_non_default_months >= {probation}
                     THEN as_of_date ELSE NULL END) OVER (
                PARTITION BY loan_id ORDER BY as_of_date
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ) AS cure_date_candidate
        FROM streaks
    ),
    default_rows AS (
        SELECT *
        FROM rows_with_cure_candidate
        WHERE default_flag
    ),
    default_rows_with_prior_cure AS (
        SELECT
            d.*,
            (
                SELECT max(p.as_of_date)
                FROM default_rows p
                WHERE p.loan_id = d.loan_id
                  AND p.as_of_date < d.as_of_date
            ) AS previous_default_date,
            (
                SELECT max(p.cure_date_candidate)
                FROM default_rows p
                WHERE p.loan_id = d.loan_id
                  AND p.as_of_date < d.as_of_date
                  AND p.cure_date_candidate < d.as_of_date
            ) AS latest_cure_before_default
        FROM default_rows d
    ),
    episode_entries AS (
        SELECT
            *,
            previous_default_date IS NULL
            OR latest_cure_before_default > previous_default_date AS is_episode_entry
        FROM default_rows_with_prior_cure
    ),
    entries AS (
        SELECT
            loan_id,
            vintage_year,
            as_of_date AS default_entry_date,
            cure_date_candidate AS cure_date,
            default_reason,
            default_due_to_delinquency,
            default_due_to_credit_event,
            row_number() OVER (PARTITION BY loan_id ORDER BY as_of_date) AS default_episode_id
        FROM episode_entries
        WHERE is_episode_entry
    )
    SELECT
        loan_id,
        vintage_year,
        default_episode_id,
        default_entry_date,
        cure_date,
        default_reason,
        default_due_to_delinquency,
        default_due_to_credit_event,
        default_episode_id > 1 AS redefault_flag,
        CASE WHEN default_episode_id > 1 THEN default_entry_date ELSE NULL END AS redefault_date,
        '{datetime.now(UTC).isoformat()}' AS _target_processed_at
    FROM entries
    """


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
    commit = commit_result.stdout.strip()
    status = status_result.stdout.strip()
    return {"commit": commit, "dirty": str(bool(status))}


def _write_default_summary(con: duckdb.DuckDBPyConnection, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
          WITH rows AS (
            SELECT 'observations' AS metric, year(as_of_date)::VARCHAR AS year,
                   NULL AS reason, NULL AS event_type, COUNT(*) AS count, NULL AS rate
            FROM pd_12m_targets GROUP BY 2
            UNION ALL
            SELECT 'eligible_12m_observations', year(as_of_date)::VARCHAR,
                   NULL, NULL, COUNT(*), NULL
            FROM pd_12m_targets WHERE eligible_for_12m_pd GROUP BY 2
            UNION ALL
            SELECT 'observable_12m_targets', year(as_of_date)::VARCHAR,
                   NULL, NULL, COUNT(*), NULL
            FROM pd_12m_targets WHERE target_12m_observable GROUP BY 2
            UNION ALL
            SELECT 'censored_12m_targets', year(as_of_date)::VARCHAR,
                   NULL, NULL, COUNT(*), NULL
            FROM pd_12m_targets
            WHERE eligible_for_12m_pd AND NOT target_12m_observable GROUP BY 2
            UNION ALL
            SELECT 'defaults', year(first_default_date)::VARCHAR,
                   default_reason, NULL, COUNT(DISTINCT loan_id), NULL
            FROM pd_12m_targets
            WHERE as_of_date = first_default_date
            GROUP BY 2, 3
            UNION ALL
            SELECT 'cures', year(cure_date)::VARCHAR,
                   default_reason, NULL, COUNT(*), NULL
            FROM default_events
            WHERE cure_date IS NOT NULL
            GROUP BY 2, 3
            UNION ALL
            SELECT 'redefaults', year(redefault_date)::VARCHAR,
                   default_reason, NULL, COUNT(*), NULL
            FROM default_events
            WHERE redefault_flag
            GROUP BY 2, 3
            UNION ALL
            SELECT 'non_default_exits', year(future_first_non_default_exit_date)::VARCHAR,
                   future_first_non_default_exit_reason, event_type, COUNT(DISTINCT loan_id), NULL
            FROM pd_12m_targets
            WHERE event_prepayment_or_exit
            GROUP BY 2, 3, 4
          )
          SELECT * FROM rows
          ORDER BY metric, year, reason, event_type
        )
        TO '{(artifact_dir / "default_summary.csv").as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )


def _write_target_observability(con: duckdb.DuckDBPyConnection, artifact_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT
              year(as_of_date) AS observation_year,
              COUNT(*) AS observations,
              SUM(CASE WHEN eligible_for_12m_pd THEN 1 ELSE 0 END) AS eligible_observations,
              SUM(CASE WHEN target_12m_observable THEN 1 ELSE 0 END) AS observable_12m_targets,
              SUM(CASE WHEN eligible_for_12m_pd AND NOT target_12m_observable
                       THEN 1 ELSE 0 END) AS censored_12m_targets,
              SUM(CASE WHEN default_next_12m = 1 THEN 1 ELSE 0 END) AS positive_12m_targets,
              SUM(CASE WHEN default_next_12m = 0 THEN 1 ELSE 0 END) AS negative_12m_targets,
              SUM(CASE WHEN default_next_12m = 1 THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN target_12m_observable THEN 1 ELSE 0 END), 0)
                AS observable_12m_bad_rate
          FROM pd_12m_targets
          GROUP BY 1
          ORDER BY 1
        )
        TO '{(artifact_dir / "target_observability.csv").as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )


def _write_manifest(
    artifact_dir: Path,
    result: TargetBuildResult,
    config: DefaultDefinitionConfig,
    git_state: dict[str, str],
    validation: dict[str, object],
) -> None:
    payload = {
        "status": "success",
        "build_timestamp": datetime.now(UTC).isoformat(),
        "default_definition_version": config.version,
        "reporting_cutoff": config.reporting_cutoff.isoformat(),
        "delinquency_threshold_months": config.delinquency_threshold_months,
        "target_horizon_months": config.target_horizon_months,
        "cure_probation_months": config.cure.probation_months,
        "git": git_state,
        "result": asdict(result),
        "validation": validation,
    }
    (artifact_dir / "targets_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _validate_targets(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    row = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT loan_id || '|' || CAST(as_of_date AS VARCHAR)) AS unique_keys,
            SUM(CASE WHEN eligible_for_12m_pd THEN 1 ELSE 0 END) AS eligible_12m,
            SUM(CASE WHEN default_next_12m = 1 THEN 1 ELSE 0 END) AS positive_12m,
            SUM(CASE WHEN default_next_12m = 0 THEN 1 ELSE 0 END) AS negative_12m,
            SUM(CASE WHEN eligible_for_12m_pd AND NOT target_12m_observable
                     THEN 1 ELSE 0 END) AS censored_12m
        FROM pd_12m_targets
        """
    ).fetchone()
    events = con.execute(
        """
        SELECT
            COUNT(*) AS default_event_rows,
            COUNT(DISTINCT loan_id) AS loans_with_default_events,
            SUM(CASE WHEN cure_date IS NOT NULL THEN 1 ELSE 0 END) AS cures,
            SUM(CASE WHEN redefault_flag THEN 1 ELSE 0 END) AS redefaults
        FROM default_events
        """
    ).fetchone()
    return {
        "pd_12m_rows": row[0],
        "pd_12m_unique_keys": row[1],
        "eligible_12m_observations": row[2],
        "positive_12m_targets": row[3],
        "negative_12m_targets": row[4],
        "censored_12m_targets": row[5],
        "default_event_rows": events[0],
        "loans_with_default_events": events[1],
        "cures": events[2],
        "redefaults": events[3],
    }


def run_target_build(
    *,
    all_targets: bool = False,
    force: bool = False,
    repo_root: Path | None = None,
) -> TargetBuildResult:
    """Build default events and point-in-time PD target tables."""
    configure_logging()
    if not all_targets:
        msg = "Specify all_targets=True"
        raise TargetBuildError(msg)
    root = find_repo_root(repo_root)
    data_config = _load_data_config(root)
    default_config = load_default_definition(root)
    gold_root = root / data_config["ingestion"]["gold_freddie_dir"]
    targets_root = gold_root / "targets"
    artifact_dir = root / "artifacts" / "targets"
    pd_dir = targets_root / "pd_12m_targets"
    events_dir = targets_root / "default_events"
    if (pd_dir.exists() or events_dir.exists()) and not force:
        msg = "Target outputs already exist; rerun with --force to overwrite"
        raise TargetBuildError(msg)

    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    LOGGER.info("Building default events and PD targets")
    pd_tmp = _prepare_tmp(pd_dir)
    events_tmp = _prepare_tmp(events_dir)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        with duckdb.connect(database=":memory:") as con:
            con.execute("PRAGMA threads=4")
            register_gold_views(con, root)
            _copy_query(con, _target_sql(default_config), pd_tmp / "part-pd_12m_targets.parquet")
            _copy_query(
                con,
                _default_events_sql(default_config),
                events_tmp / "part-default_events.parquet",
            )
            _promote_tmp(pd_tmp, pd_dir)
            _promote_tmp(events_tmp, events_dir)
            con.execute(
                f"""
                CREATE OR REPLACE VIEW pd_12m_targets AS
                SELECT * FROM read_parquet('{pd_dir.as_posix()}/*.parquet')
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE VIEW default_events AS
                SELECT * FROM read_parquet('{events_dir.as_posix()}/*.parquet')
                """
            )
            _write_default_summary(con, artifact_dir)
            _write_target_observability(con, artifact_dir)
            validation = _validate_targets(con)
    except Exception:
        for tmp in [pd_tmp, events_tmp]:
            if tmp.exists():
                shutil.rmtree(tmp)
        raise

    result = TargetBuildResult(
        pd_12m_rows=validation["pd_12m_rows"],
        default_event_rows=validation["default_event_rows"],
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        status="success",
        processing_time_seconds=time.perf_counter() - start,
    )
    _write_manifest(artifact_dir, result, default_config, _git_state(root), validation)
    LOGGER.info("Default definition and PD target build completed")
    return result
