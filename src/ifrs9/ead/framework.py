"""IFRS 9 EAD population, target, profile, and backtesting framework."""

from __future__ import annotations

import json
import pickle
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ifrs9.ead.config import EADFrameworkConfig, load_ead_config
from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.views import register_gold_views
from ifrs9.targets.exposure import default_entry_ead_source_sql, default_entry_ead_sql

EAD_OUTCOME_EXCLUDED_COLUMNS = {
    "ead_at_default",
    "ead_at_default_source",
    "ead_ratio",
    "future_default_date",
    "future_non_default_exit_date",
    "months_to_default",
    "event_type",
    "target_observable",
    "predicted_ead_current_balance",
    "predicted_ead_contractual",
    "predicted_ead_empirical_model",
}
METHOD_COLUMNS = {
    "current_balance": "predicted_ead_current_balance",
    "contractual_amortization": "predicted_ead_contractual",
    "empirical_model": "predicted_ead_empirical_model",
}


@dataclass(frozen=True)
class EADRunResult:
    """Result of an EAD framework run."""

    run_id: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


def run_ead_framework(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> EADRunResult:
    """Run the configurable EAD framework."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_ead_config(root, config_path)
    run_id = run_id or f"ead_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.run.artifact_root / run_id
    model_dir = root / config.run.model_root / run_id
    if not force and (artifact_dir.exists() or model_dir.exists()):
        msg = f"EAD run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    observations = build_ead_observation_dataset(root, config)
    observations = _add_splits(observations, config)
    predictors = _validated_predictors(config)
    model, scored = _fit_empirical_model(observations, predictors, config)
    scored = _score_baseline_methods(scored)
    profiles = _build_ead_profiles(scored, model, predictors, config)

    backtest = _backtest_by_split(scored, config)
    segment_backtest = _segment_backtests(scored, config)
    ratio_distribution = _ead_ratio_distribution(scored)
    population_summary = _population_summary(scored)
    profile_validation = _profile_validation(profiles)
    target_reconciliation = _target_reconciliation(scored)

    scored.to_parquet(artifact_dir / "ead_observations.parquet", index=False)
    profiles.to_parquet(artifact_dir / "ead_profiles.parquet", index=False)
    backtest.to_csv(artifact_dir / "backtest_by_split.csv", index=False)
    segment_backtest.to_csv(artifact_dir / "backtest_by_segment.csv", index=False)
    ratio_distribution.to_csv(artifact_dir / "ead_ratio_distribution.csv", index=False)
    population_summary.to_csv(artifact_dir / "population_summary.csv", index=False)
    profile_validation.to_csv(artifact_dir / "profile_validation.csv", index=False)
    target_reconciliation.to_csv(artifact_dir / "target_reconciliation.csv", index=False)
    _write_json(artifact_dir / "config_snapshot.json", config.model_dump())
    with (model_dir / "ead_model.pkl").open("wb") as stream:
        pickle.dump({"model": model, "predictors": predictors}, stream)

    result = EADRunResult(
        run_id=run_id,
        artifact_path=str(artifact_dir),
        model_path=str(model_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "config": config.model_dump(),
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
        "target_dataset_metadata": {
            "rows": len(scored),
            "observable_defaults": int(scored["target_observable"].sum()),
            "censored_or_non_default_exits": int((~scored["target_observable"]).sum()),
            "predictors": predictors,
            "outcome_excluded_columns": sorted(EAD_OUTCOME_EXCLUDED_COLUMNS),
        },
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


def build_ead_observation_dataset(repo_root: Path, config: EADFrameworkConfig) -> pd.DataFrame:
    """Build eligible point-in-time EAD observations from the Gold mart."""
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        default_events = repo_root / "data" / "gold" / "freddie" / "targets" / "default_events"
        pd_targets = repo_root / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
        pd_predictions = (
            repo_root / "artifacts" / "pd" / config.parent_pd_run / "pd_predictions.parquet"
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW default_events AS
            SELECT * FROM read_parquet('{default_events.as_posix()}/*.parquet')
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
            CREATE OR REPLACE VIEW pd_predictions AS
            SELECT * FROM read_parquet('{pd_predictions.as_posix()}')
            """
        )
        return con.execute(_observation_sql(config)).fetchdf()


def _observation_sql(config: EADFrameworkConfig) -> str:
    ead_expression = default_entry_ead_sql("dm")
    ead_source_expression = default_entry_ead_source_sql("dm")
    non_default_limit = config.target_population.max_non_default_exit_observations
    include_non_default = config.target_population.include_non_default_exits
    non_default_predicate = (
        "OR (t.event_prepayment_or_exit AND t.non_default_exit_sample_rank <= "
        f"{non_default_limit})"
        if include_non_default and non_default_limit
        else "OR t.event_prepayment_or_exit"
        if include_non_default
        else ""
    )
    return f"""
    WITH ranked_targets AS (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY event_prepayment_or_exit
                ORDER BY as_of_date, loan_id
            ) AS non_default_exit_sample_rank
        FROM pd_12m_targets
        WHERE eligible_for_lifetime_pd
    ),
    base AS (
        SELECT
            m.loan_id,
            m.vintage_year,
            m.as_of_date,
            m.current_actual_upb,
            m.current_interest_rate,
            COALESCE(
                NULLIF(m.remaining_months_to_maturity_source, 0),
                NULLIF(m.months_to_contractual_maturity, 0)
            ) AS remaining_months_to_maturity,
            m.months_to_contractual_maturity,
            m.original_loan_term AS original_term,
            m.original_upb,
            m.current_upb_to_original_upb,
            m.estimated_loan_to_value,
            m.delinquency_months,
            m.max_delinquency_months_to_date,
            m.months_delinquent_to_date,
            m.months_since_last_delinquency,
            m.months_since_origination,
            m.current_non_interest_bearing_upb,
            m.current_interest_bearing_upb,
            m.modification_flag,
            m.ever_modified_to_date,
            m.ever_assistance_to_date,
            m.current_assistance_flag,
            m.property_state,
            m.property_type,
            m.occupancy_status,
            m.loan_purpose,
            m.channel,
            m.zero_balance_code,
            CASE
                WHEN m.estimated_loan_to_value IS NULL THEN 'UNKNOWN'
                WHEN m.estimated_loan_to_value <= 60 THEN '<=60'
                WHEN m.estimated_loan_to_value <= 80 THEN '60-80'
                WHEN m.estimated_loan_to_value <= 100 THEN '80-100'
                ELSE '>100'
            END AS current_ltv_band,
            CASE
                WHEN m.current_actual_upb <= 100000 THEN '<=100k'
                WHEN m.current_actual_upb <= 250000 THEN '100k-250k'
                WHEN m.current_actual_upb <= 500000 THEN '250k-500k'
                ELSE '>500k'
            END AS balance_band
        FROM gold_loan_month_analytical m
        JOIN ranked_targets t
          ON m.loan_id = t.loan_id
         AND m.vintage_year = t.vintage_year
         AND m.as_of_date = t.as_of_date
        WHERE m.current_actual_upb > 0
          AND NULLIF(m.zero_balance_code, '') IS NULL
          AND (
            t.event_default
            {non_default_predicate}
          )
    ),
    default_ead AS (
        SELECT
            e.loan_id,
            e.vintage_year,
            e.default_episode_id,
            e.default_entry_date AS default_date,
            e.default_reason,
            {ead_expression} AS ead_at_default,
            {ead_source_expression} AS ead_at_default_source
        FROM default_events e
        LEFT JOIN gold_loan_month dm
          ON e.loan_id = dm.loan_id
         AND e.vintage_year = dm.vintage_year
         AND e.default_entry_date = dm.as_of_date
    ),
    rating_lookup AS (
        SELECT
            b.loan_id,
            b.vintage_year,
            b.as_of_date,
            p.rating,
            row_number() OVER (
                PARTITION BY b.loan_id, b.vintage_year, b.as_of_date
                ORDER BY p.as_of_date DESC
            ) AS rating_rank
        FROM base b
        LEFT JOIN pd_predictions p
          ON b.loan_id = p.loan_id
         AND p.as_of_date <= b.as_of_date
    )
    SELECT
        b.*,
        r.rating,
        de.default_episode_id,
        CASE WHEN t.event_default THEN t.first_default_date ELSE NULL END
            AS future_default_date,
        de.default_reason,
        CASE WHEN t.event_default THEN de.ead_at_default ELSE NULL END
            AS ead_at_default,
        CASE WHEN t.event_default THEN de.ead_at_default_source ELSE NULL END
            AS ead_at_default_source,
        t.future_first_non_default_exit_date,
        t.future_first_non_default_exit_reason AS future_non_default_exit_code,
        CASE
            WHEN t.event_default THEN t.months_to_default_from_observation
            ELSE NULL
        END AS months_to_default,
        CASE WHEN t.event_default THEN 'DEFAULT' ELSE t.event_type END AS event_type,
        false AS ever_default_to_date
    FROM base b
    JOIN ranked_targets t
      ON b.loan_id = t.loan_id
     AND b.vintage_year = t.vintage_year
     AND b.as_of_date = t.as_of_date
    LEFT JOIN default_ead de
      ON b.loan_id = de.loan_id
     AND b.vintage_year = de.vintage_year
     AND t.first_default_date = de.default_date
    LEFT JOIN rating_lookup r
      ON b.loan_id = r.loan_id
     AND b.vintage_year = r.vintage_year
     AND b.as_of_date = r.as_of_date
     AND r.rating_rank = 1
    """


def _add_splits(frame: pd.DataFrame, config: EADFrameworkConfig) -> pd.DataFrame:
    output = frame.copy()
    dates = pd.to_datetime(output["as_of_date"])
    output["observation_year"] = dates.dt.year
    output["split"] = np.select(
        [
            dates <= pd.Timestamp(config.development.train_end),
            dates <= pd.Timestamp(config.development.validation_end),
            dates <= pd.Timestamp(config.development.oot_end),
        ],
        ["TRAIN", "VALIDATION", "OOT"],
        default="OUT_OF_TIME",
    )
    output["target_observable"] = (
        output["event_type"].eq("DEFAULT")
        & output["ead_at_default"].notna()
        & output["current_actual_upb"].gt(0)
    )
    output["months_to_horizon"] = output["months_to_default"]
    output["ead_ratio"] = np.where(
        output["target_observable"],
        output["ead_at_default"] / output["current_actual_upb"],
        np.nan,
    )
    output["ead_ratio_model_target"] = output["ead_ratio"].clip(
        config.empirical_model.lower_bound,
        config.empirical_model.upper_bound,
    )
    output["months_to_default_band"] = pd.cut(
        output["months_to_default"],
        bins=[0, 3, 6, 12, 24, 60, np.inf],
        labels=["0-3", "4-6", "7-12", "13-24", "25-60", "60+"],
        include_lowest=True,
        right=True,
    ).astype("string")
    return output


def _validated_predictors(config: EADFrameworkConfig) -> list[str]:
    predictors = config.features.numeric + config.features.categorical
    forbidden = sorted(set(predictors).intersection(EAD_OUTCOME_EXCLUDED_COLUMNS))
    if forbidden:
        msg = f"Outcome fields are not allowed as EAD predictors: {forbidden}"
        raise ValueError(msg)
    return predictors


def _fit_empirical_model(
    observations: pd.DataFrame,
    predictors: list[str],
    config: EADFrameworkConfig,
) -> tuple[Pipeline, pd.DataFrame]:
    scored = observations.copy()
    target_rows = scored[scored["target_observable"]].copy()
    train = target_rows["split"] == "TRAIN"
    model = _model_pipeline(config.features.numeric, config.features.categorical)
    model.fit(
        target_rows.loc[train, predictors],
        target_rows.loc[train, "ead_ratio_model_target"],
    )
    ratio = model.predict(scored[predictors])
    scored["predicted_ead_ratio_empirical"] = np.clip(
        ratio,
        config.empirical_model.lower_bound,
        config.empirical_model.upper_bound,
    )
    scored["predicted_ead_empirical_model"] = (
        scored["predicted_ead_ratio_empirical"] * scored["current_actual_upb"]
    ).clip(lower=0)
    return model, scored


def _model_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("features", transformer), ("model", LinearRegression())])


def _score_baseline_methods(scored: pd.DataFrame) -> pd.DataFrame:
    output = scored.copy()
    output["predicted_ead_current_balance"] = output["current_actual_upb"].clip(lower=0)
    output["predicted_ead_contractual"] = contractual_balance(
        output["current_actual_upb"],
        output["current_interest_rate"],
        output["remaining_months_to_maturity"],
        output["months_to_default"].fillna(0),
    )
    return output


def contractual_balance(
    current_upb: pd.Series | np.ndarray | float,
    annual_rate_pct: pd.Series | np.ndarray | float,
    remaining_months: pd.Series | np.ndarray | float,
    horizon_month: pd.Series | np.ndarray | float,
) -> np.ndarray:
    """Project amortizing mortgage balance at a future monthly horizon."""
    balance = np.asarray(current_upb, dtype=float)
    rate = np.maximum(np.asarray(annual_rate_pct, dtype=float), 0) / 100 / 12
    maturity = np.maximum(np.asarray(remaining_months, dtype=float), 0)
    horizon = np.maximum(np.asarray(horizon_month, dtype=float), 0)
    horizon = np.minimum(horizon, maturity)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        payment = np.where(
            rate > 0,
            balance * rate / (1 - np.power(1 + rate, -maturity)),
            np.where(maturity > 0, balance / maturity, balance),
        )
        powered = np.power(1 + rate, horizon)
        amortized = np.where(
            rate > 0,
            balance * powered - payment * ((powered - 1) / rate),
            balance - payment * horizon,
        )
    amortized = np.where(horizon >= maturity, 0, amortized)
    amortized = np.where(maturity <= 0, 0, amortized)
    return np.maximum(np.nan_to_num(amortized, nan=0.0, posinf=balance), 0)


def _build_ead_profiles(
    scored: pd.DataFrame,
    model: Pipeline,
    predictors: list[str],
    config: EADFrameworkConfig,
) -> pd.DataFrame:
    profile_base = scored[scored["current_actual_upb"].gt(0)].copy()
    if config.horizons.profile_population == "latest_per_loan":
        profile_base = (
            profile_base.sort_values("as_of_date")
            .groupby("loan_id", as_index=False)
            .tail(1)
        )
    if config.horizons.max_profile_observations is not None:
        profile_base = profile_base.sort_values(["as_of_date", "loan_id"]).head(
            config.horizons.max_profile_observations
        )
    months = np.arange(config.horizons.max_months + 1)
    repeated = profile_base.loc[profile_base.index.repeat(len(months))].copy()
    repeated["month"] = np.tile(months, len(profile_base))
    repeated["months_to_horizon"] = repeated["month"]
    repeated["ead_current_balance"] = repeated["current_actual_upb"].clip(lower=0)
    repeated["ead_contractual"] = contractual_balance(
        repeated["current_actual_upb"],
        repeated["current_interest_rate"],
        repeated["remaining_months_to_maturity"],
        repeated["month"],
    )
    ratio = model.predict(repeated[predictors])
    repeated["ead_modelled"] = (
        np.clip(ratio, config.empirical_model.lower_bound, config.empirical_model.upper_bound)
        * repeated["current_actual_upb"]
    )
    repeated.loc[repeated["month"] == 0, "ead_modelled"] = repeated.loc[
        repeated["month"] == 0,
        "current_actual_upb",
    ]
    output_columns = [
        "loan_id",
        "vintage_year",
        "as_of_date",
        "month",
        "ead_current_balance",
        "ead_contractual",
        "ead_modelled",
    ]
    output = repeated[output_columns].copy()
    ead_columns = ["ead_current_balance", "ead_contractual", "ead_modelled"]
    output[ead_columns] = output[ead_columns].clip(lower=0)
    return output


def _backtest_by_split(scored: pd.DataFrame, config: EADFrameworkConfig) -> pd.DataFrame:
    rows = []
    defaults = scored[scored["target_observable"]]
    for method, prediction_column in METHOD_COLUMNS.items():
        for split, group in defaults.groupby("split"):
            rows.append(_metric_row(group, method, split, prediction_column, config))
    return pd.DataFrame(rows)


def _segment_backtests(scored: pd.DataFrame, config: EADFrameworkConfig) -> pd.DataFrame:
    rows = []
    defaults = scored[scored["target_observable"]].copy()
    segment_columns = [
        ("months_to_default_band", "months_to_default"),
        ("rating", "rating"),
        ("observation_year", "observation_year"),
        ("current_ltv_band", "current_ltv_band"),
        ("balance_band", "balance_band"),
    ]
    for method, prediction_column in METHOD_COLUMNS.items():
        for column, segment_name in segment_columns:
            for (split, value), group in defaults.groupby(["split", column], observed=True):
                row = _metric_row(group, method, split, prediction_column, config)
                row["segment"] = segment_name
                row["segment_value"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(
    group: pd.DataFrame,
    method: str,
    split: str,
    prediction_column: str,
    config: EADFrameworkConfig,
) -> dict[str, Any]:
    actual = group["ead_at_default"]
    predicted = group[prediction_column]
    safe = actual.abs() >= config.reporting.mape_denominator_floor
    return {
        "method": method,
        "split": split,
        "rows": len(group),
        "actual_mean_ead": actual.mean(),
        "predicted_mean_ead": predicted.mean(),
        "oe_ratio": actual.mean() / predicted.mean() if predicted.mean() else np.nan,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mape": (np.abs(actual[safe] - predicted[safe]) / actual[safe]).mean()
        if safe.any()
        else np.nan,
    }


def _ead_ratio_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    defaults = scored[scored["target_observable"]]
    return pd.DataFrame(
        [
            {
                "observable_defaults": len(defaults),
                "ead_ratio_mean": defaults["ead_ratio"].mean(),
                "ead_ratio_median": defaults["ead_ratio"].median(),
                "ead_ratio_p05": defaults["ead_ratio"].quantile(0.05),
                "ead_ratio_p95": defaults["ead_ratio"].quantile(0.95),
                "ead_ratio_below_0_pct": (defaults["ead_ratio"] < 0).mean(),
                "ead_ratio_above_1_pct": (defaults["ead_ratio"] > 1).mean(),
                "ead_ratio_above_2_pct": (defaults["ead_ratio"] > 2).mean(),
            }
        ]
    )


def _population_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["split", "event_type"], observed=True)
        .agg(
            observations=("loan_id", "size"),
            loans=("loan_id", "nunique"),
            observable_defaults=("target_observable", "sum"),
            mean_current_upb=("current_actual_upb", "mean"),
        )
        .reset_index()
    )


def _profile_validation(profiles: pd.DataFrame) -> pd.DataFrame:
    month_zero = profiles[profiles["month"] == 0]
    return pd.DataFrame(
        [
            {
                "profile_rows": len(profiles),
                "month_zero_rows": len(month_zero),
                "month_zero_current_equals_contractual": (
                    month_zero["ead_current_balance"].round(6)
                    == month_zero["ead_contractual"].round(6)
                ).mean(),
                "month_zero_current_equals_modelled": (
                    month_zero["ead_current_balance"].round(6)
                    == month_zero["ead_modelled"].round(6)
                ).mean(),
                "negative_ead_rows": int(
                    (
                        profiles[
                            ["ead_current_balance", "ead_contractual", "ead_modelled"]
                        ]
                        < 0
                    )
                    .any(axis=1)
                    .sum()
                ),
            }
        ]
    )


def _target_reconciliation(scored: pd.DataFrame) -> pd.DataFrame:
    defaults = scored[scored["target_observable"]]
    return (
        defaults.groupby("ead_at_default_source", observed=True)
        .agg(
            rows=("loan_id", "size"),
            actual_mean_ead=("ead_at_default", "mean"),
            actual_sum_ead=("ead_at_default", "sum"),
        )
        .reset_index()
    )


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


def load_ead_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted EAD run manifest."""
    config = load_ead_config(repo_root)
    return json.loads((repo_root / config.run.artifact_root / run_id / "run.json").read_text())
