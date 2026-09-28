"""IFRS 9 LGD population, target, modelling, and backtesting framework."""

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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.lgd.config import LGDFrameworkConfig, load_lgd_config
from ifrs9.mart.views import register_gold_views

RECOVERY_COMPONENTS = [
    "net_sale_proceeds",
    "mi_recoveries",
    "non_mi_recoveries",
]
COST_COMPONENTS = [
    "total_expenses",
    "delinquent_accrued_interest",
    "cumulative_modification_costs",
    "bankruptcy_cramdown_costs",
]
OUTCOME_EXCLUDED_COLUMNS = set(RECOVERY_COMPONENTS + COST_COMPONENTS) | {
    "actual_loss",
    "resolution_type",
    "resolution_date",
    "months_to_resolution",
    "resolved_flag",
    "cured_flag",
    "realized_lgd_raw",
    "realized_lgd_model_target",
    "economic_loss",
    "discounted_recoveries",
    "discounted_costs",
}


@dataclass(frozen=True)
class LGDRunResult:
    """Result of an LGD framework run."""

    run_id: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


def run_lgd_framework(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> LGDRunResult:
    """Run the configurable LGD framework."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_lgd_config(root, config_path)
    run_id = run_id or f"lgd_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.run.artifact_root / run_id
    model_dir = root / config.run.model_root / run_id
    if not force and (artifact_dir.exists() or model_dir.exists()):
        msg = f"LGD run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    episodes = build_lgd_episode_dataset(root, config)
    episodes = _add_lgd_targets(episodes, config)
    episodes = _add_splits(episodes, config)
    predictors = _validated_predictors(config)
    models, scored, metrics = _fit_models(episodes, predictors, config)
    downturn = _downturn_table(scored, config)
    scored = scored.merge(
        downturn[["rating_at_default", "downturn_overlay_factor"]],
        on="rating_at_default",
        how="left",
    )
    scored["downturn_overlay_factor"] = scored["downturn_overlay_factor"].fillna(1.0)
    scored["lgd_base"] = scored["predicted_lgd"]
    scored["lgd_downturn"] = np.clip(scored["lgd_base"] * scored["downturn_overlay_factor"], 0, 1)

    backtesting_split = _backtesting_by_split(scored)
    by_rating = _observed_predicted(scored, ["split", "rating_at_default"])
    by_year = _observed_predicted(scored, ["split", "default_year"])
    by_resolution = _observed_predicted(scored, ["split", "resolution_type"])
    recovery_timing = _recovery_timing(scored)
    resolution_distribution = _resolution_distribution(scored)
    lgd_diagnostics = _lgd_diagnostics(scored)
    reconciliation = _actual_loss_reconciliation(scored)
    segmentation = _segmentation_report(scored, config)

    scored.to_parquet(artifact_dir / "lgd_episodes.parquet", index=False)
    pd.DataFrame(metrics).to_csv(artifact_dir / "model_metrics.csv", index=False)
    backtesting_split.to_csv(artifact_dir / "backtesting_split.csv", index=False)
    by_rating.to_csv(artifact_dir / "observed_predicted_by_rating.csv", index=False)
    by_year.to_csv(artifact_dir / "observed_predicted_by_default_year.csv", index=False)
    by_resolution.to_csv(artifact_dir / "observed_predicted_by_resolution_type.csv", index=False)
    recovery_timing.to_csv(artifact_dir / "recovery_timing.csv", index=False)
    resolution_distribution.to_csv(artifact_dir / "resolution_distribution.csv", index=False)
    lgd_diagnostics.to_csv(artifact_dir / "lgd_diagnostics.csv", index=False)
    reconciliation.to_csv(artifact_dir / "actual_loss_reconciliation.csv", index=False)
    downturn.to_csv(artifact_dir / "downturn_overlay.csv", index=False)
    segmentation.to_csv(artifact_dir / "segmentation_report.csv", index=False)
    _cashflow_component_summary(scored).to_csv(
        artifact_dir / "cashflow_component_summary.csv",
        index=False,
    )
    _write_json(artifact_dir / "config_snapshot.json", config.model_dump())
    with (model_dir / "lgd_models.pkl").open("wb") as stream:
        pickle.dump({"models": models, "predictors": predictors}, stream)

    result = LGDRunResult(
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
            "resolved": int(scored["resolved_flag"].sum()),
            "unresolved": int((~scored["resolved_flag"]).sum()),
            "predictors": predictors,
            "outcome_excluded_columns": sorted(OUTCOME_EXCLUDED_COLUMNS),
        },
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


def build_lgd_episode_dataset(repo_root: Path, config: LGDFrameworkConfig) -> pd.DataFrame:
    """Build one LGD row per existing default episode."""
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        default_events = repo_root / "data" / "gold" / "freddie" / "targets" / "default_events"
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
            CREATE OR REPLACE VIEW pd_predictions AS
            SELECT * FROM read_parquet('{pd_predictions.as_posix()}')
            """
        )
        return con.execute(_episode_sql(config)).fetchdf()


def _episode_sql(config: LGDFrameworkConfig) -> str:
    horizon = config.resolution_horizon_months
    return f"""
    WITH entries AS (
        SELECT
            e.loan_id,
            e.vintage_year,
            e.default_episode_id,
            e.default_entry_date AS default_date,
            e.cure_date,
            e.default_reason,
            e.default_due_to_delinquency,
            e.default_due_to_credit_event,
            e.redefault_flag,
            COALESCE(NULLIF(m.current_actual_upb, 0), m.current_upb_lag_1) AS ead_at_default,
            CASE
                WHEN m.current_actual_upb > 0 THEN 'current_actual_upb'
                WHEN m.current_upb_lag_1 > 0 THEN 'prior_month_current_upb'
                ELSE 'unavailable'
            END AS ead_at_default_source,
            m.current_interest_rate AS current_interest_rate_at_default,
            m.estimated_loan_to_value AS estimated_loan_to_value_at_default,
            m.current_upb_to_original_upb AS current_upb_to_original_upb_at_default,
            m.delinquency_months AS delinquency_months_at_default,
            m.max_delinquency_months_to_date AS max_delinquency_months_to_date_at_default,
            m.months_delinquent_to_date AS months_delinquent_to_date_at_default,
            m.months_since_last_delinquency AS months_since_last_delinquency_at_default,
            m.months_since_origination AS months_since_origination_at_default,
            m.ever_modified_to_date AS ever_modified_to_date_at_default,
            m.ever_assistance_to_date AS ever_assistance_to_date_at_default,
            s.original_credit_score,
            s.original_dti,
            s.original_ltv,
            s.original_cltv,
            s.original_upb,
            s.original_interest_rate,
            s.mortgage_insurance_percentage,
            s.property_state,
            s.property_type,
            s.occupancy_status,
            s.loan_purpose,
            s.channel
        FROM default_events e
        LEFT JOIN gold_loan_month m
          ON e.loan_id = m.loan_id
         AND e.vintage_year = m.vintage_year
         AND e.default_entry_date = m.as_of_date
        LEFT JOIN gold_loan_static s
          ON e.loan_id = s.loan_id
         AND e.vintage_year = s.vintage_year
    ),
    terminal_rows AS (
        SELECT
            e.loan_id,
            e.default_episode_id,
            m.as_of_date AS terminal_date,
            m.zero_balance_code,
            m.current_actual_upb,
            TRY_CAST(NULLIF(m.net_sale_proceeds, 'U') AS DOUBLE) AS net_sale_proceeds,
            m.mi_recoveries,
            m.non_mi_recoveries,
            m.total_expenses,
            m.legal_costs,
            m.maintenance_and_preservation_costs,
            m.taxes_and_insurance,
            m.miscellaneous_expenses,
            m.actual_loss,
            m.cumulative_modification_costs,
            m.current_period_modification_costs,
            m.bankruptcy_cramdown_costs,
            CAST(NULL AS DOUBLE) AS delinquent_accrued_interest,
            row_number() OVER (
                PARTITION BY e.loan_id, e.default_episode_id
                ORDER BY m.as_of_date
            ) AS terminal_rank
        FROM entries e
        JOIN gold_loan_month m
          ON e.loan_id = m.loan_id
         AND e.vintage_year = m.vintage_year
         AND m.as_of_date >= e.default_date
         AND m.as_of_date <= e.default_date + INTERVAL {horizon} MONTH
         AND NULLIF(m.zero_balance_code, '') IS NOT NULL
    ),
    first_terminal AS (
        SELECT * EXCLUDE (terminal_rank)
        FROM terminal_rows
        WHERE terminal_rank = 1
    ),
    rating_lookup AS (
        SELECT
            e.loan_id,
            e.default_episode_id,
            p.rating,
            row_number() OVER (
                PARTITION BY e.loan_id, e.default_episode_id
                ORDER BY p.as_of_date DESC
            ) AS rating_rank
        FROM entries e
        LEFT JOIN pd_predictions p
          ON e.loan_id = p.loan_id
         AND p.as_of_date <= e.default_date
    ),
    resolved AS (
        SELECT
            e.*,
            r.rating AS rating_at_default,
            t.terminal_date,
            t.zero_balance_code,
            t.net_sale_proceeds,
            t.mi_recoveries,
            t.non_mi_recoveries,
            t.total_expenses,
            t.legal_costs,
            t.maintenance_and_preservation_costs,
            t.taxes_and_insurance,
            t.miscellaneous_expenses,
            t.actual_loss,
            t.cumulative_modification_costs,
            t.current_period_modification_costs,
            t.bankruptcy_cramdown_costs,
            t.delinquent_accrued_interest,
            CASE
                WHEN e.cure_date IS NOT NULL
                 AND (t.terminal_date IS NULL OR e.cure_date <= t.terminal_date)
                    THEN e.cure_date
                ELSE t.terminal_date
            END AS resolution_date,
            CASE
                WHEN e.cure_date IS NOT NULL
                 AND (t.terminal_date IS NULL OR e.cure_date <= t.terminal_date)
                    THEN 'CURE'
                WHEN t.zero_balance_code = '09' THEN 'REO_FORECLOSURE'
                WHEN t.zero_balance_code = '03' THEN 'SHORT_SALE_OR_CHARGE_OFF'
                WHEN t.zero_balance_code = '02' THEN 'THIRD_PARTY_SALE'
                WHEN t.zero_balance_code IN ('01', '15', '16', '96')
                    THEN 'OTHER_DOCUMENTED_TERMINAL'
                ELSE 'UNRESOLVED'
            END AS resolution_type
        FROM entries e
        LEFT JOIN first_terminal t
          ON e.loan_id = t.loan_id
         AND e.default_episode_id = t.default_episode_id
        LEFT JOIN rating_lookup r
          ON e.loan_id = r.loan_id
         AND e.default_episode_id = r.default_episode_id
         AND r.rating_rank = 1
    )
    SELECT
        *,
        resolution_type <> 'UNRESOLVED' AS resolved_flag,
        resolution_type = 'CURE' AS cured_flag,
        CASE
            WHEN resolution_type = 'UNRESOLVED' THEN NULL
            ELSE date_diff('month', default_date, resolution_date)
        END AS months_to_resolution,
        CASE
            WHEN estimated_loan_to_value_at_default IS NULL THEN 'UNKNOWN'
            WHEN estimated_loan_to_value_at_default <= 60 THEN '<=60'
            WHEN estimated_loan_to_value_at_default <= 80 THEN '60-80'
            WHEN estimated_loan_to_value_at_default <= 100 THEN '80-100'
            ELSE '>100'
        END AS current_ltv_band
    FROM resolved
    """


def _add_lgd_targets(frame: pd.DataFrame, config: LGDFrameworkConfig) -> pd.DataFrame:
    output = frame.copy()
    rate = output["current_interest_rate_at_default"].copy()
    if config.discounting.fallback_rate is not None:
        rate = rate.fillna(config.discounting.fallback_rate)
    rate = rate.fillna(output["original_interest_rate"]).fillna(0.0) / 100 / 12
    months = output["months_to_resolution"].fillna(0).clip(lower=0)
    discount_factor = np.power(1 + rate, -months)
    recoveries = output[RECOVERY_COMPONENTS].fillna(0).sum(axis=1)
    costs = output[COST_COMPONENTS].fillna(0).sum(axis=1)
    observed_cashflow_amount = (
        output[RECOVERY_COMPONENTS + COST_COMPONENTS].fillna(0).abs().sum(axis=1)
    )
    cured_flag = output.get("cured_flag", pd.Series(False, index=output.index))
    actual_loss = output.get("actual_loss", pd.Series(np.nan, index=output.index))
    output["cure_lgd_fallback_used"] = (
        cured_flag & (observed_cashflow_amount == 0) & actual_loss.isna()
    )
    output["discount_factor_to_resolution"] = discount_factor
    output["discounted_recoveries"] = recoveries * discount_factor
    output["discounted_costs"] = costs * discount_factor
    output["economic_loss"] = (
        output["ead_at_default"].fillna(0)
        - output["discounted_recoveries"]
        + output["discounted_costs"]
    )
    output.loc[output["cure_lgd_fallback_used"], "economic_loss"] = (
        output.loc[output["cure_lgd_fallback_used"], "ead_at_default"]
        * config.target.cure_fallback_lgd
    )
    output["realized_lgd_raw"] = np.where(
        output["ead_at_default"] > 0,
        output["economic_loss"] / output["ead_at_default"],
        np.nan,
    )
    output.loc[~output["resolved_flag"], "realized_lgd_raw"] = np.nan
    output["realized_lgd_model_target"] = output["realized_lgd_raw"].clip(
        config.target.lower_bound,
        config.target.upper_bound,
    )
    return output


def _add_splits(frame: pd.DataFrame, config: LGDFrameworkConfig) -> pd.DataFrame:
    output = frame.copy()
    dates = pd.to_datetime(output["default_date"])
    output["default_year"] = dates.dt.year
    output["split"] = np.select(
        [
            dates <= pd.Timestamp(config.development.train_end),
            dates <= pd.Timestamp(config.development.validation_end),
            dates <= pd.Timestamp(config.development.oot_end),
        ],
        ["TRAIN", "VALIDATION", "OOT"],
        default="OUT_OF_TIME",
    )
    return output


def _validated_predictors(config: LGDFrameworkConfig) -> list[str]:
    predictors = config.features.numeric + config.features.categorical
    forbidden = sorted(set(predictors).intersection(OUTCOME_EXCLUDED_COLUMNS))
    if forbidden:
        msg = f"Outcome fields are not allowed as LGD predictors: {forbidden}"
        raise ValueError(msg)
    return predictors


def _fit_models(
    episodes: pd.DataFrame,
    predictors: list[str],
    config: LGDFrameworkConfig,
) -> tuple[dict[str, Pipeline], pd.DataFrame, list[dict[str, Any]]]:
    scored = episodes.copy()
    supervised = scored[scored["resolved_flag"] & scored["ead_at_default"].gt(0)].copy()
    train = supervised["split"] == "TRAIN"
    cure_model = _model_pipeline(config.features.numeric, config.features.categorical, "logistic")
    cure_model.fit(
        supervised.loc[train, predictors],
        supervised.loc[train, "cured_flag"].astype(int),
    )
    scored["predicted_cure_probability"] = cure_model.predict_proba(scored[predictors])[:, 1]

    non_cure = supervised[~supervised["cured_flag"]].copy()
    severity_train = non_cure["split"] == "TRAIN"
    severity_model = _model_pipeline(config.features.numeric, config.features.categorical, "linear")
    severity_model.fit(
        non_cure.loc[severity_train, predictors],
        non_cure.loc[severity_train, "realized_lgd_model_target"],
    )
    scored["predicted_non_cure_lgd"] = np.clip(severity_model.predict(scored[predictors]), 0, 1)
    cure_loss = float(
        supervised.loc[supervised["cured_flag"], "realized_lgd_model_target"].mean()
        if supervised["cured_flag"].any()
        else 0.0
    )
    scored["predicted_cure_lgd"] = cure_loss
    scored["predicted_lgd"] = (
        scored["predicted_cure_probability"] * scored["predicted_cure_lgd"]
        + (1 - scored["predicted_cure_probability"]) * scored["predicted_non_cure_lgd"]
    )
    metrics = _model_metrics(scored)
    return {"cure_model": cure_model, "severity_model": severity_model}, scored, metrics


def _model_pipeline(numeric: list[str], categorical: list[str], kind: str) -> Pipeline:
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
    estimator = (
        LogisticRegression(max_iter=500, class_weight="balanced")
        if kind == "logistic"
        else LinearRegression()
    )
    return Pipeline([("features", transformer), ("model", estimator)])


def _model_metrics(scored: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split, group in scored[scored["resolved_flag"]].groupby("split"):
        if group.empty:
            continue
        target = group["realized_lgd_model_target"]
        rows.append(
            {
                "model": "expected_lgd",
                "split": split,
                "rows": len(group),
                "mae": mean_absolute_error(target, group["predicted_lgd"]),
                "rmse": mean_squared_error(target, group["predicted_lgd"]) ** 0.5,
            }
        )
        if group["cured_flag"].nunique() > 1:
            rows.append(
                {
                    "model": "cure",
                    "split": split,
                    "rows": len(group),
                    "roc_auc": roc_auc_score(
                        group["cured_flag"].astype(int),
                        group["predicted_cure_probability"],
                    ),
                    "observed_cure_rate": group["cured_flag"].mean(),
                    "predicted_cure_rate": group["predicted_cure_probability"].mean(),
                }
            )
        non_cure = group[~group["cured_flag"]]
        if not non_cure.empty:
            rows.append(
                {
                    "model": "non_cure_severity",
                    "split": split,
                    "rows": len(non_cure),
                    "mae": mean_absolute_error(
                        non_cure["realized_lgd_model_target"],
                        non_cure["predicted_non_cure_lgd"],
                    ),
                    "rmse": mean_squared_error(
                        non_cure["realized_lgd_model_target"],
                        non_cure["predicted_non_cure_lgd"],
                    )
                    ** 0.5,
                }
            )
    return rows


def _backtesting_by_split(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split"):
        resolved = group[group["resolved_flag"]]
        rows.append(
            {
                "split": split,
                "default_episodes": len(group),
                "resolved_defaults": len(resolved),
                "unresolved_defaults": int((~group["resolved_flag"]).sum()),
                "cure_rate": resolved["cured_flag"].mean() if len(resolved) else np.nan,
                "realized_mean_lgd": resolved["realized_lgd_model_target"].mean(),
                "realized_median_lgd": resolved["realized_lgd_model_target"].median(),
                "predicted_lgd": resolved["predicted_lgd"].mean(),
                "mae": mean_absolute_error(
                    resolved["realized_lgd_model_target"],
                    resolved["predicted_lgd"],
                )
                if len(resolved)
                else np.nan,
                "rmse": mean_squared_error(
                    resolved["realized_lgd_model_target"],
                    resolved["predicted_lgd"],
                )
                ** 0.5
                if len(resolved)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _observed_predicted(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]]
    if resolved.empty:
        return pd.DataFrame()
    return (
        resolved.groupby(keys, observed=True)
        .agg(
            resolved_defaults=("loan_id", "size"),
            realized_lgd=("realized_lgd_model_target", "mean"),
            predicted_lgd=("predicted_lgd", "mean"),
        )
        .reset_index()
    )


def _recovery_timing(scored: pd.DataFrame) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]]
    return (
        resolved.groupby(["split", "resolution_type"], observed=True)["months_to_resolution"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )


def _resolution_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["resolution_type"], observed=True)
        .agg(
            default_episodes=("loan_id", "size"),
            realized_lgd=("realized_lgd_model_target", "mean"),
        )
        .reset_index()
    )


def _lgd_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]]
    return pd.DataFrame(
        [
            {
                "resolved_defaults": len(resolved),
                "raw_lgd_mean": resolved["realized_lgd_raw"].mean(),
                "raw_lgd_median": resolved["realized_lgd_raw"].median(),
                "raw_lgd_below_0_pct": (resolved["realized_lgd_raw"] < 0).mean(),
                "raw_lgd_above_1_pct": (resolved["realized_lgd_raw"] > 1).mean(),
                "raw_lgd_below_0_amount": resolved.loc[
                    resolved["realized_lgd_raw"] < 0,
                    "realized_lgd_raw",
                ].sum(),
                "raw_lgd_above_1_amount": resolved.loc[
                    resolved["realized_lgd_raw"] > 1,
                    "realized_lgd_raw",
                ].sum(),
            }
        ]
    )


def _actual_loss_reconciliation(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored[scored["actual_loss"].notna() & scored["resolved_flag"]].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["economic_minus_actual_loss"] = frame["economic_loss"] - frame["actual_loss"]
    return pd.DataFrame(
        [
            {
                "rows_with_actual_loss": len(frame),
                "actual_loss_sum": frame["actual_loss"].sum(),
                "economic_loss_sum": frame["economic_loss"].sum(),
                "difference_sum": frame["economic_minus_actual_loss"].sum(),
                "difference_mean": frame["economic_minus_actual_loss"].mean(),
                "correlation": frame["economic_loss"].corr(frame["actual_loss"]),
            }
        ]
    )


def _downturn_table(scored: pd.DataFrame, config: LGDFrameworkConfig) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]].copy()
    if config.downturn.method == "none" or resolved.empty:
        ratings = scored["rating_at_default"].dropna().unique()
        return pd.DataFrame({"rating_at_default": ratings, "downturn_overlay_factor": 1.0})
    if config.downturn.method == "historical_stress":
        stress = resolved[
            (pd.to_datetime(resolved["default_date"]) >= pd.Timestamp(config.downturn.stress_start))
            & (pd.to_datetime(resolved["default_date"]) <= pd.Timestamp(config.downturn.stress_end))
        ]
        baseline = _mean_lgd_by_rating(resolved)
        stress_mean = _mean_lgd_by_rating(stress)
        factor = _overlay_factor(stress_mean, baseline)
    else:
        baseline = _mean_lgd_by_rating(resolved)
        stressed = (
            resolved.groupby("rating_at_default", observed=True)["realized_lgd_model_target"]
            .quantile(config.downturn.quantile)
        )
        factor = _overlay_factor(stressed, baseline)
    return factor.rename("downturn_overlay_factor").reset_index()


def _mean_lgd_by_rating(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("rating_at_default", observed=True)["realized_lgd_model_target"].mean()


def _overlay_factor(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        (numerator / denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=1.0)
    )


def _segmentation_report(scored: pd.DataFrame, config: LGDFrameworkConfig) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]]
    rows = []
    for variable in config.segmentation.variables:
        for value, group in resolved.groupby(variable, observed=True):
            rows.append(
                {
                    "segment_variable": variable,
                    "segment_value": value,
                    "resolved_defaults": len(group),
                    "accepted_segment": len(group) >= config.segmentation.minimum_resolved_defaults,
                    "realized_lgd": group["realized_lgd_model_target"].mean(),
                    "predicted_lgd": group["predicted_lgd"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _cashflow_component_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in RECOVERY_COMPONENTS + COST_COMPONENTS + ["actual_loss"]:
        rows.append(
            {
                "component": column,
                "non_null_rows": int(scored[column].notna().sum()),
                "sum": scored[column].fillna(0).sum(),
                "mean_non_null": scored[column].mean(),
            }
        )
    return pd.DataFrame(rows)


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


def load_lgd_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted LGD run manifest."""
    config = load_lgd_config(repo_root)
    return json.loads((repo_root / config.run.artifact_root / run_id / "run.json").read_text())
