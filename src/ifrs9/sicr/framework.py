"""IFRS 9 SICR trigger and staging framework."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.views import register_gold_views
from ifrs9.sicr.config import StagingFrameworkConfig, load_staging_config
from ifrs9.targets.exposure import default_entry_ead_sql

RATING_NOTCH = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}


@dataclass(frozen=True)
class StagingRunResult:
    """Result of a staging framework run."""

    run_id: str
    staging_path: str
    artifact_path: str
    processing_time_seconds: float


@dataclass(frozen=True)
class StagingScenarioResult:
    """Result of a SICR threshold scenario simulation."""

    scenario_id: str
    stage2_count: int
    stage2_ead: float
    migration_rate: float


def run_staging(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> StagingRunResult:
    """Run the configurable SICR and staging framework."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_staging_config(root, config_path)
    run_id = run_id or f"sicr_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    staging_dir = root / config.output.staging_path
    artifact_dir = root / config.output.artifact_root / run_id
    if not force and artifact_dir.exists():
        msg = f"SICR run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if staging_dir.exists() and force:
        shutil.rmtree(staging_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    base = build_staging_base_dataset(root, config)
    staged = allocate_stages(base, config)
    scenarios = simulate_sicr_thresholds(config, staged)

    staged[_staging_output_columns()].to_parquet(staging_dir / "part-staging.parquet", index=False)
    _stage_distribution(staged).to_csv(artifact_dir / "stage_distribution.csv", index=False)
    _stage_distribution_by_year_rating(staged).to_csv(
        artifact_dir / "stage_distribution_by_year_rating.csv",
        index=False,
    )
    _trigger_distribution(staged).to_csv(artifact_dir / "trigger_distribution.csv", index=False)
    _stage_migrations(staged).to_csv(artifact_dir / "stage_migrations.csv", index=False)
    scenarios.to_csv(artifact_dir / "threshold_sensitivity.csv", index=False)
    _reference_diagnostics(staged).to_csv(
        artifact_dir / "reference_pd_diagnostics.csv",
        index=False,
    )
    _stage3_state_audit(staged).to_csv(artifact_dir / "stage3_state_audit.csv", index=False)
    _sicr_trigger_exclusivity(staged).to_csv(
        artifact_dir / "sicr_trigger_exclusivity.csv",
        index=False,
    )
    _pd_relative_change_distribution(staged).to_csv(
        artifact_dir / "pd_relative_change_distribution.csv",
        index=False,
    )
    _reference_lag_profile(staged).to_csv(artifact_dir / "reference_lag_profile.csv", index=False)
    _write_json(artifact_dir / "config_snapshot.json", config.model_dump())

    result = StagingRunResult(
        run_id=run_id,
        staging_path=str(staging_dir),
        artifact_path=str(artifact_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "config": config.model_dump(),
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
        "rows": len(staged),
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


def build_staging_base_dataset(repo_root: Path, config: StagingFrameworkConfig) -> pd.DataFrame:
    """Build the compact point-in-time staging population from PD and Gold outputs."""
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        pd_predictions = (
            repo_root / "artifacts" / "pd" / config.parent_pd_run / "pd_predictions.parquet"
        )
        pd_targets = repo_root / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
        default_events = repo_root / "data" / "gold" / "freddie" / "targets" / "default_events"
        con.execute(
            f"""
            CREATE OR REPLACE VIEW pd_predictions AS
            SELECT * FROM read_parquet('{pd_predictions.as_posix()}')
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW pd_12m_targets AS
            SELECT * FROM read_parquet('{pd_targets.as_posix()}/*.parquet')
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW default_events AS
            SELECT * FROM read_parquet('{default_events.as_posix()}/*.parquet')
            """
        )
        return con.execute(_staging_base_sql()).fetchdf()


def _staging_base_sql() -> str:
    default_ead = default_entry_ead_sql("m")
    return f"""
    WITH pd_scored_rows AS (
        SELECT
            p.loan_id,
            p.as_of_date,
            p.pd_calibrated_12m AS pd_current,
            p.rating AS rating_current,
            m.vintage_year,
            CASE
                WHEN e.default_entry_date = p.as_of_date THEN {default_ead}
                ELSE m.current_actual_upb
            END AS ead_current,
            m.delinquency_months,
            m.ever_modified_to_date,
            m.ever_assistance_to_date,
            m.current_assistance_flag,
            m.as_of_date AS mart_as_of_date,
            s.origination_date,
            COALESCE(t.default_flag, false) AS default_entry_event,
            e.default_episode_id,
            e.default_entry_date,
            e.cure_date,
            e.default_reason,
            e.default_episode_id IS NOT NULL AS active_default_state
        FROM pd_predictions p
        LEFT JOIN gold_loan_month m
          ON p.loan_id = m.loan_id
         AND p.as_of_date = m.as_of_date
        LEFT JOIN gold_loan_static s
          ON m.loan_id = s.loan_id
         AND m.vintage_year = s.vintage_year
        LEFT JOIN pd_12m_targets t
          ON p.loan_id = t.loan_id
         AND m.vintage_year = t.vintage_year
         AND p.as_of_date = t.as_of_date
        LEFT JOIN default_events e
          ON p.loan_id = e.loan_id
         AND m.vintage_year = e.vintage_year
         AND p.as_of_date >= e.default_entry_date
         AND (e.cure_date IS NULL OR p.as_of_date < e.cure_date)
        WHERE m.loan_id IS NOT NULL
    ),
    active_default_months AS (
        SELECT
            e.loan_id,
            m.as_of_date,
            e.vintage_year,
            CASE
                WHEN m.as_of_date = e.default_entry_date THEN {default_ead}
                ELSE m.current_actual_upb
            END AS ead_current,
            m.delinquency_months,
            m.ever_modified_to_date,
            m.ever_assistance_to_date,
            m.current_assistance_flag,
            m.as_of_date AS mart_as_of_date,
            s.origination_date,
            e.default_episode_id,
            e.default_entry_date,
            e.cure_date,
            e.default_reason
        FROM default_events e
        INNER JOIN gold_loan_month m
          ON e.loan_id = m.loan_id
         AND e.vintage_year = m.vintage_year
         AND m.as_of_date >= e.default_entry_date
         AND (e.cure_date IS NULL OR m.as_of_date < e.cure_date)
        LEFT JOIN gold_loan_static s
          ON e.loan_id = s.loan_id
         AND e.vintage_year = s.vintage_year
        LEFT JOIN pd_predictions p
          ON e.loan_id = p.loan_id
         AND m.as_of_date = p.as_of_date
        WHERE p.loan_id IS NULL
    ),
    active_default_rating_lookup AS (
        SELECT
            a.loan_id,
            a.vintage_year,
            a.as_of_date,
            p.pd_calibrated_12m AS pd_current,
            p.rating AS rating_current,
            row_number() OVER (
                PARTITION BY a.loan_id, a.vintage_year, a.as_of_date
                ORDER BY p.as_of_date DESC
            ) AS rating_rank
        FROM active_default_months a
        LEFT JOIN pd_predictions p
          ON a.loan_id = p.loan_id
         AND p.as_of_date <= a.as_of_date
    ),
    active_default_rows AS (
        SELECT
            a.loan_id,
            a.as_of_date,
            r.pd_current,
            r.rating_current,
            a.vintage_year,
            a.ead_current,
            a.delinquency_months,
            a.ever_modified_to_date,
            a.ever_assistance_to_date,
            a.current_assistance_flag,
            a.mart_as_of_date,
            a.origination_date,
            a.as_of_date = a.default_entry_date AS default_entry_event,
            a.default_episode_id,
            a.default_entry_date,
            a.cure_date,
            a.default_reason,
            true AS active_default_state
        FROM active_default_months a
        LEFT JOIN active_default_rating_lookup r
          ON a.loan_id = r.loan_id
         AND a.vintage_year = r.vintage_year
         AND a.as_of_date = r.as_of_date
         AND r.rating_rank = 1
    ),
    pd_base AS (
        SELECT * FROM pd_scored_rows
        UNION ALL
        SELECT * FROM active_default_rows
    ),
    reference AS (
        SELECT
            loan_id,
            as_of_date AS sicr_reference_date,
            pd_current AS sicr_reference_pd,
            rating_current AS sicr_reference_rating,
            row_number() OVER (PARTITION BY loan_id ORDER BY as_of_date) AS reference_rank
        FROM pd_base
        WHERE pd_current IS NOT NULL
          AND rating_current IS NOT NULL
    )
    SELECT
        b.*,
        r.sicr_reference_date,
        r.sicr_reference_pd,
        r.sicr_reference_rating,
        CASE
            WHEN b.origination_date IS NULL OR r.sicr_reference_date IS NULL THEN NULL
            ELSE date_diff('month', b.origination_date, r.sicr_reference_date)
        END AS months_origination_to_reference
    FROM pd_base b
    LEFT JOIN reference r
      ON b.loan_id = r.loan_id
     AND r.reference_rank = 1
    ORDER BY b.loan_id, b.as_of_date
    """


def allocate_stages(frame: pd.DataFrame, config: StagingFrameworkConfig) -> pd.DataFrame:
    """Apply SICR triggers, cure rules, and stage allocation."""
    output = frame.copy()
    output["pd_origination"] = output["sicr_reference_pd"]
    output["pd_current"] = output["pd_current"].astype(float)
    output["pd_relative_change"] = np.where(
        output["pd_origination"].gt(0),
        output["pd_current"] / output["pd_origination"],
        np.nan,
    )
    output["pd_absolute_change"] = output["pd_current"] - output["pd_origination"]
    output["rating_origination"] = output["sicr_reference_rating"]
    output["rating_current"] = output["rating_current"]
    output["rating_notch_change"] = (
        output["rating_current"].map(RATING_NOTCH)
        - output["rating_origination"].map(RATING_NOTCH)
    )
    reference_date = pd.to_datetime(output["sicr_reference_date"])
    as_of_date = pd.to_datetime(output["as_of_date"])
    output["origination_baseline_available"] = output["pd_origination"].notna() & (
        reference_date <= as_of_date
    )
    if "default_entry_event" not in output:
        output["default_entry_event"] = output.get("default_flag", False)
    if "active_default_state" not in output:
        output["active_default_state"] = output["default_entry_event"]
    output["default_entry_event"] = output["default_entry_event"].fillna(False).astype(bool)
    output["active_default_state"] = output["active_default_state"].fillna(False).astype(bool)
    output["stage3_flag"] = (
        output["active_default_state"]
        if config.stage3.default_flag
        else False
    )

    low_credit_risk = _low_credit_risk_flag(output, config)
    output["low_credit_risk_exemption"] = low_credit_risk
    output["sicr_relative_pd"] = (
        output["origination_baseline_available"]
        & output["pd_relative_change"].ge(config.sicr.relative_pd_increase)
        if config.sicr.relative_pd_increase is not None
        else False
    )
    output["sicr_absolute_pd"] = (
        output["origination_baseline_available"]
        & output["pd_absolute_change"].ge(config.sicr.absolute_pd_increase)
        if config.sicr.absolute_pd_increase is not None
        else False
    )
    output["sicr_rating_downgrade"] = (
        output["rating_notch_change"].ge(config.sicr.rating_downgrade_notches)
        if config.sicr.rating_downgrade_notches is not None
        else False
    )
    output["sicr_dpd_backstop"] = (
        output["delinquency_months"].fillna(0).ge(config.sicr.dpd_backstop_months)
        if config.sicr.dpd_backstop_months is not None
        else False
    )
    output["sicr_other_credit_deterioration"] = _other_deterioration_flag(output, config)
    trigger_columns = _trigger_columns()
    output[trigger_columns] = output[trigger_columns].fillna(False).astype(bool)
    output.loc[low_credit_risk, trigger_columns] = False
    output["sicr_raw_flag"] = output[trigger_columns].any(axis=1)
    output["sicr_flag"] = _apply_stage2_cure(output, config)
    output["sicr_reason"] = _reason(output, trigger_columns, "NO_SICR")
    output["stage"] = np.select(
        [output["stage3_flag"], output["sicr_flag"]],
        [3, 2],
        default=1,
    )
    output["stage_reason"] = np.select(
        [output["stage3_flag"], output["sicr_flag"]],
        ["DEFAULT", output["sicr_reason"]],
        default="NO_SICR",
    )
    return output


def simulate_sicr_thresholds(
    config: StagingFrameworkConfig,
    staged: pd.DataFrame,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run deterministic SICR threshold sensitivity without refitting PD."""
    rows = []
    rel_values = overrides.get("relative_pd_increase") if overrides else None
    rating_values = overrides.get("rating_downgrade_notches") if overrides else None
    dpd_values = overrides.get("dpd_backstop_months") if overrides else None
    rel_values = rel_values or config.sensitivity.relative_pd_increase
    rating_values = rating_values or config.sensitivity.rating_downgrade_notches
    dpd_values = dpd_values or config.sensitivity.dpd_backstop_months
    for relative_pd in rel_values:
        for rating_notches in rating_values:
            for dpd_months in dpd_values:
                stage2 = (
                    staged["origination_baseline_available"]
                    & staged["pd_relative_change"].ge(relative_pd)
                )
                stage2 |= staged["rating_notch_change"].ge(rating_notches)
                stage2 |= staged["delinquency_months"].fillna(0).ge(dpd_months)
                stage2 |= staged["sicr_other_credit_deterioration"]
                stage2 &= ~staged["low_credit_risk_exemption"]
                stage2 &= ~staged["stage3_flag"]
                stage = np.select([staged["stage3_flag"], stage2], [3, 2], default=1)
                migration_rate = _migration_rate(staged["loan_id"], staged["as_of_date"], stage)
                rows.append(
                    {
                        "relative_pd_increase": relative_pd,
                        "rating_downgrade_notches": rating_notches,
                        "dpd_backstop_months": dpd_months,
                        "stage2_count": int((stage == 2).sum()),
                        "stage2_ead": staged.loc[stage == 2, "ead_current"].fillna(0).sum(),
                        "migration_rate": migration_rate,
                    }
                )
    return pd.DataFrame(rows)


def simulate_threshold_stage_distribution(
    config: StagingFrameworkConfig,
    staged: pd.DataFrame,
    overrides: dict[str, Any],
) -> pd.DataFrame:
    """Simulate one SICR threshold setting and return stage counts and EAD."""
    prepared = staged.copy()
    if "stage3_flag" not in prepared:
        prepared["stage3_flag"] = prepared["stage"].eq(3)
    if "low_credit_risk_exemption" not in prepared:
        prepared["low_credit_risk_exemption"] = False
    if "sicr_other_credit_deterioration" not in prepared:
        prepared["sicr_other_credit_deterioration"] = False
    relative_pd = overrides.get("relative_pd_increase", config.sicr.relative_pd_increase)
    rating_notches = overrides.get(
        "rating_downgrade_notches",
        config.sicr.rating_downgrade_notches,
    )
    dpd_months = overrides.get("dpd_backstop_months", config.sicr.dpd_backstop_months)
    absolute_pd = overrides.get("absolute_pd_increase", config.sicr.absolute_pd_increase)
    stage2 = pd.Series(False, index=prepared.index)
    if relative_pd is not None:
        stage2 |= (
            prepared["origination_baseline_available"].fillna(False)
            & prepared["pd_relative_change"].ge(relative_pd)
        )
    if absolute_pd is not None:
        stage2 |= (
            prepared["origination_baseline_available"].fillna(False)
            & prepared["pd_absolute_change"].ge(absolute_pd)
        )
    if rating_notches is not None:
        stage2 |= prepared["rating_notch_change"].ge(rating_notches)
    if dpd_months is not None:
        stage2 |= prepared["sicr_dpd_backstop"].fillna(False)
        stage2 |= prepared.get("delinquency_months", pd.Series(0, index=prepared.index)).fillna(
            0
        ).ge(dpd_months)
    stage2 |= prepared["sicr_other_credit_deterioration"].fillna(False)
    stage2 &= ~prepared["low_credit_risk_exemption"].fillna(False)
    stage2 &= ~prepared["stage3_flag"].fillna(False)
    stage = np.select([prepared["stage3_flag"], stage2], [3, 2], default=1)
    prepared["simulated_stage"] = stage
    return (
        prepared.groupby("simulated_stage", observed=True)
        .agg(rows=("loan_id", "size"), loans=("loan_id", "nunique"), ead=("ead_current", "sum"))
        .reset_index()
        .rename(columns={"simulated_stage": "stage"})
    )


def _apply_stage2_cure(frame: pd.DataFrame, config: StagingFrameworkConfig) -> pd.Series:
    if config.cure.stage2_to_1_rule == "immediate" or config.cure.probation_months == 0:
        return frame["sicr_raw_flag"] & ~frame["stage3_flag"]
    ordered = frame.sort_values(["loan_id", "as_of_date"]).copy()
    dates = pd.to_datetime(ordered["as_of_date"])
    previous_date = dates.groupby(ordered["loan_id"]).shift()
    month_index = dates.dt.year * 12 + dates.dt.month
    previous_month_index = previous_date.dt.year * 12 + previous_date.dt.month
    month_gap = (month_index - previous_month_index).fillna(1).ne(1)
    raw_sicr_for_cure = ordered["sicr_raw_flag"] & ~ordered["stage3_flag"]
    reset = raw_sicr_for_cure | ordered["stage3_flag"] | month_gap
    group_id = reset.groupby(ordered["loan_id"]).cumsum()
    no_sicr = ~(ordered["sicr_raw_flag"] | ordered["stage3_flag"])
    no_sicr_run = no_sicr.astype(int).groupby([ordered["loan_id"], group_id]).cumsum()
    group_has_sicr = raw_sicr_for_cure.groupby([ordered["loan_id"], group_id]).transform("max")
    adjusted = (
        ordered["sicr_raw_flag"]
        | (group_has_sicr & no_sicr & no_sicr_run.lt(config.cure.probation_months))
    )
    adjusted &= ~ordered["stage3_flag"]
    return adjusted.reindex(frame.index).fillna(False).astype(bool)


def _low_credit_risk_flag(frame: pd.DataFrame, config: StagingFrameworkConfig) -> pd.Series:
    if not config.low_credit_risk.enabled or config.low_credit_risk.max_rating is None:
        return pd.Series(False, index=frame.index)
    max_notch = RATING_NOTCH.get(config.low_credit_risk.max_rating)
    if max_notch is None:
        return pd.Series(False, index=frame.index)
    return frame["rating_current"].map(RATING_NOTCH).le(max_notch).fillna(False)


def _other_deterioration_flag(frame: pd.DataFrame, config: StagingFrameworkConfig) -> pd.Series:
    controls = config.sicr.other_credit_deterioration
    if not controls.enabled:
        return pd.Series(False, index=frame.index)
    flags = pd.Series(False, index=frame.index)
    if controls.include_modification:
        flags |= frame["ever_modified_to_date"].fillna(False).astype(bool)
    if controls.include_assistance:
        flags |= (
            frame["ever_assistance_to_date"].fillna(False).astype(bool)
            | frame["current_assistance_flag"].fillna(False).astype(bool)
        )
    return flags


def _reason(frame: pd.DataFrame, columns: list[str], fallback: str) -> pd.Series:
    reason = pd.Series("", index=frame.index)
    for column in columns:
        reason = reason.mask(frame[column], reason + "|" + column)
    reason = reason.str.strip("|")
    return reason.replace("", fallback)


def _trigger_columns() -> list[str]:
    return [
        "sicr_relative_pd",
        "sicr_absolute_pd",
        "sicr_rating_downgrade",
        "sicr_dpd_backstop",
        "sicr_other_credit_deterioration",
    ]


def _staging_output_columns() -> list[str]:
    return [
        "loan_id",
        "as_of_date",
        "ead_current",
        "pd_origination",
        "pd_current",
        "pd_relative_change",
        "pd_absolute_change",
        "rating_origination",
        "rating_current",
        "rating_notch_change",
        "default_entry_event",
        "active_default_state",
        "default_episode_id",
        "default_entry_date",
        "cure_date",
        "default_reason",
        "sicr_reference_date",
        "sicr_reference_pd",
        "sicr_reference_rating",
        "months_origination_to_reference",
        "origination_baseline_available",
        "sicr_relative_pd",
        "sicr_absolute_pd",
        "sicr_rating_downgrade",
        "sicr_dpd_backstop",
        "sicr_other_credit_deterioration",
        "sicr_flag",
        "sicr_reason",
        "stage",
        "stage_reason",
    ]


def _stage_distribution(staged: pd.DataFrame) -> pd.DataFrame:
    return (
        staged.groupby(["stage"], observed=True)
        .agg(
            rows=("loan_id", "size"),
            loans=("loan_id", "nunique"),
            ead=("ead_current", "sum"),
            avg_pd=("pd_current", "mean"),
            avg_ead=("ead_current", "mean"),
        )
        .reset_index()
    )


def _stage_distribution_by_year_rating(staged: pd.DataFrame) -> pd.DataFrame:
    frame = staged.copy()
    frame["year"] = pd.to_datetime(frame["as_of_date"]).dt.year
    return (
        frame.groupby(["year", "rating_current", "stage"], observed=True)
        .agg(
            rows=("loan_id", "size"),
            loans=("loan_id", "nunique"),
            ead=("ead_current", "sum"),
            avg_pd=("pd_current", "mean"),
            avg_ead=("ead_current", "mean"),
        )
        .reset_index()
    )


def _trigger_distribution(staged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trigger_columns = _trigger_columns()
    for column in trigger_columns:
        rows.append(
            {
                "trigger": column,
                "rows": int(staged[column].sum()),
                "ead": staged.loc[staged[column], "ead_current"].fillna(0).sum(),
            }
        )
    overlap = staged[trigger_columns].sum(axis=1)
    rows.append(
        {
            "trigger": "multiple_trigger_overlap",
            "rows": int((overlap > 1).sum()),
            "ead": staged.loc[overlap > 1, "ead_current"].fillna(0).sum(),
        }
    )
    return pd.DataFrame(rows)


def _stage3_state_audit(staged: pd.DataFrame) -> pd.DataFrame:
    active = staged[staged["active_default_state"]].copy()
    if active.empty:
        return pd.DataFrame(
            [
                {
                    "metric": "stage3_entries",
                    "value": int(staged["default_entry_event"].sum()),
                    "ead": staged.loc[staged["default_entry_event"], "ead_current"].sum(),
                },
                {"metric": "stage3_active_rows", "value": 0, "ead": 0.0},
            ]
        )
    episode_cols = ["loan_id", "default_episode_id", "default_entry_date", "cure_date"]
    episode_summary = (
        active.groupby(episode_cols, dropna=False, observed=True)
        .agg(active_month_rows=("loan_id", "size"), ead=("ead_current", "sum"))
        .reset_index()
    )
    cured = episode_summary["cure_date"].notna()
    rows = [
        {
            "metric": "stage3_entries",
            "value": int(staged["default_entry_event"].sum()),
            "ead": staged.loc[staged["default_entry_event"], "ead_current"].sum(),
        },
        {
            "metric": "stage3_active_rows",
            "value": int(staged["active_default_state"].sum()),
            "ead": active["ead_current"].sum(),
        },
        {
            "metric": "average_months_in_stage3",
            "value": float(episode_summary["active_month_rows"].mean()),
            "ead": np.nan,
        },
        {
            "metric": "median_months_in_stage3",
            "value": float(episode_summary["active_month_rows"].median()),
            "ead": np.nan,
        },
        {"metric": "cured_stage3_episodes", "value": int(cured.sum()), "ead": np.nan},
        {
            "metric": "unresolved_or_terminal_stage3_episodes",
            "value": int((~cured).sum()),
            "ead": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def _sicr_trigger_exclusivity(staged: pd.DataFrame) -> pd.DataFrame:
    trigger_columns = _trigger_columns()
    frame = staged.copy()
    trigger_count = frame[trigger_columns].sum(axis=1)
    rows = []
    for column in trigger_columns:
        mask = frame[column] & trigger_count.eq(1)
        rows.append(
            {
                "trigger_set": f"only_{column}",
                "rows": int(mask.sum()),
                "ead": frame.loc[mask, "ead_current"].fillna(0).sum(),
            }
        )
    rows.append(
        {
            "trigger_set": "overlap_two_or_more",
            "rows": int(trigger_count.gt(1).sum()),
            "ead": frame.loc[trigger_count.gt(1), "ead_current"].fillna(0).sum(),
        }
    )
    rows.append(
        {
            "trigger_set": "no_stage2_trigger",
            "rows": int(trigger_count.eq(0).sum()),
            "ead": frame.loc[trigger_count.eq(0), "ead_current"].fillna(0).sum(),
        }
    )
    combinations = (
        frame.groupby(trigger_columns, observed=True)
        .agg(rows=("loan_id", "size"), ead=("ead_current", "sum"))
        .reset_index()
    )
    combinations["trigger_set"] = "none"
    for column in trigger_columns:
        prefix = np.where(
            combinations["trigger_set"].eq("none"),
            "",
            combinations["trigger_set"] + "+",
        )
        combinations["trigger_set"] = combinations["trigger_set"].mask(
            combinations[column],
            prefix + column,
        )
    combinations = combinations[["trigger_set", "rows", "ead"]]
    return pd.concat([pd.DataFrame(rows), combinations], ignore_index=True)


def _pd_relative_change_distribution(staged: pd.DataFrame) -> pd.DataFrame:
    valid = staged["pd_relative_change"].replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = [
        {
            "metric": "unique_pd_current_values",
            "bucket": "all",
            "value": staged["pd_current"].nunique(),
        },
        {
            "metric": "unique_active_rating_grades",
            "bucket": "all",
            "value": staged["rating_current"].dropna().nunique(),
        },
    ]
    for quantile in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]:
        rows.append(
            {
                "metric": "pd_relative_change_quantile",
                "bucket": f"q{quantile:.2f}",
                "value": float(valid.quantile(quantile)) if not valid.empty else np.nan,
            }
        )
    bins = [0, 1, 1.5, 2, 3, 5, 10, np.inf]
    labels = ["0-1x", "1-1.5x", "1.5-2x", "2-3x", "3-5x", "5-10x", ">10x"]
    binned = pd.cut(valid, bins=bins, labels=labels, include_lowest=True, right=False)
    for bucket, count in binned.value_counts(sort=False).items():
        rows.append(
            {
                "metric": "pd_relative_change_bucket_rows",
                "bucket": str(bucket),
                "value": int(count),
            }
        )
    return pd.DataFrame(rows)


def _reference_lag_profile(staged: pd.DataFrame) -> pd.DataFrame:
    frame = staged.copy()
    lag = frame["months_origination_to_reference"]
    labels = ["0m", "1m", "2-3m", "4-6m", ">6m"]
    frame["reference_lag_band"] = pd.cut(
        lag,
        bins=[-np.inf, 0, 1, 3, 6, np.inf],
        labels=labels,
        right=True,
    )
    unavailable = frame["reference_lag_band"].isna()
    rows = (
        frame.loc[~unavailable]
        .groupby("reference_lag_band", observed=False)
        .agg(rows=("loan_id", "size"), loans=("loan_id", "nunique"), ead=("ead_current", "sum"))
        .reset_index()
    )
    if unavailable.any():
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "reference_lag_band": "unavailable",
                            "rows": int(unavailable.sum()),
                            "loans": frame.loc[unavailable, "loan_id"].nunique(),
                            "ead": frame.loc[unavailable, "ead_current"].fillna(0).sum(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows


def _stage_migrations(staged: pd.DataFrame) -> pd.DataFrame:
    ordered = staged[["loan_id", "as_of_date", "stage", "ead_current"]].sort_values(
        ["loan_id", "as_of_date"]
    )
    ordered["previous_stage"] = ordered.groupby("loan_id")["stage"].shift()
    migrations = ordered.loc[ordered["previous_stage"].notna()].copy()
    migrations["migration"] = (
        "Stage "
        + migrations["previous_stage"].astype(int).astype(str)
        + "->"
        + migrations["stage"].astype(int).astype(str)
    )
    return (
        migrations.groupby("migration", observed=True)
        .agg(rows=("loan_id", "size"), ead=("ead_current", "sum"))
        .reset_index()
    )


def _reference_diagnostics(staged: pd.DataFrame) -> pd.DataFrame:
    unavailable = ~staged["origination_baseline_available"]
    return pd.DataFrame(
        [
            {
                "rows": len(staged),
                "loans": staged["loan_id"].nunique(),
                "rows_without_reference": int(unavailable.sum()),
                "loans_without_reference": staged.loc[unavailable, "loan_id"].nunique(),
                "avg_months_origination_to_reference": staged[
                    "months_origination_to_reference"
                ].mean(),
                "median_months_origination_to_reference": staged[
                    "months_origination_to_reference"
                ].median(),
            }
        ]
    )


def _migration_rate(loan_id: pd.Series, as_of_date: pd.Series, stage: np.ndarray) -> float:
    frame = pd.DataFrame({"loan_id": loan_id, "as_of_date": as_of_date, "stage": stage})
    frame = frame.sort_values(["loan_id", "as_of_date"])
    previous = frame.groupby("loan_id")["stage"].shift()
    valid = previous.notna()
    if not valid.any():
        return 0.0
    return float((frame.loc[valid, "stage"] != previous[valid]).mean())


def _git_state(repo_root: Path) -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}
    return {"commit": commit, "dirty": str(bool(status))}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load_staging_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted SICR/staging run manifest."""
    config = load_staging_config(repo_root)
    return json.loads((repo_root / config.output.artifact_root / run_id / "run.json").read_text())
