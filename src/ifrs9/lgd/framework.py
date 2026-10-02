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
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
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
    "bankruptcy_cramdown_costs",
]
MODIFICATION_COST_COMPONENTS = ["cumulative_modification_costs"]
RECONCILIATION_COMPONENTS = [
    "zero_balance_removal_upb",
    "net_sale_proceeds",
    "mi_recoveries",
    "non_mi_recoveries",
    "total_expenses",
    "delinquent_accrued_interest",
]
TEMPORAL_WEIGHT_CANDIDATES = [
    {"method": "none", "decay_months": None},
    {"method": "linear_recency", "decay_months": None},
    {"method": "exponential_recency", "decay_months": 12},
    {"method": "exponential_recency", "decay_months": 24},
    {"method": "exponential_recency", "decay_months": 36},
]
OUTCOME_EXCLUDED_COLUMNS = set(
    RECOVERY_COMPONENTS + COST_COMPONENTS + MODIFICATION_COST_COMPONENTS
) | {
    "actual_loss",
    "zero_balance_removal_upb",
    "resolution_type",
    "resolution_date",
    "months_to_resolution",
    "resolved_flag",
    "cured_flag",
    "realized_lgd_raw",
    "realized_lgd_model_target",
    "economic_loss",
    "freddie_reconstructed_actual_loss",
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
    component_audit = _component_audit(scored)
    distribution_diagnostics = _distribution_diagnostics(scored)
    component_decomposition = _component_decomposition(scored)
    predictor_drift = _predictor_drift(scored, config)
    cure_model_comparison = pd.DataFrame(models["cure_model_comparison"])
    severity_model_comparison = pd.DataFrame(models["severity_model_comparison"])
    cure_lgd_model_comparison = pd.DataFrame(models["cure_lgd_model_comparison"])
    combined_backtest = pd.DataFrame(models["combined_backtest"])

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
    component_audit.to_csv(artifact_dir / "lgd_component_audit.csv", index=False)
    distribution_diagnostics.to_csv(
        artifact_dir / "lgd_distribution_diagnostics.csv",
        index=False,
    )
    component_decomposition.to_csv(artifact_dir / "component_decomposition.csv", index=False)
    predictor_drift.to_csv(artifact_dir / "predictor_drift.csv", index=False)
    cure_model_comparison.to_csv(artifact_dir / "cure_model_comparison.csv", index=False)
    cure_lgd_model_comparison.to_csv(
        artifact_dir / "cure_lgd_model_comparison.csv",
        index=False,
    )
    severity_model_comparison.to_csv(
        artifact_dir / "severity_model_comparison.csv",
        index=False,
    )
    combined_backtest.to_csv(artifact_dir / "combined_backtest.csv", index=False)
    downturn.to_csv(artifact_dir / "downturn_overlay.csv", index=False)
    segmentation.to_csv(artifact_dir / "segmentation_report.csv", index=False)
    _cashflow_component_summary(scored).to_csv(
        artifact_dir / "cashflow_component_summary.csv",
        index=False,
    )
    _cure_lgd_ltv_audit(scored).to_csv(artifact_dir / "cure_lgd_ltv_audit.csv", index=False)
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
        con.execute(
            f"""
            CREATE OR REPLACE VIEW silver_performance AS
            SELECT * FROM read_parquet(
                '{repo_root.as_posix()}/data/silver/freddie/performance/*/*.parquet'
            )
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
            sp.zero_balance_removal_upb,
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
            sp.delinquent_accrued_interest,
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
        LEFT JOIN silver_performance sp
          ON m.loan_id = sp.loan_id
         AND m.vintage_year = sp.vintage_year
         AND m.as_of_date = sp.period
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
            t.zero_balance_removal_upb,
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
    signed_recoveries = output[RECOVERY_COMPONENTS].fillna(0).sum(axis=1)
    costs = output[COST_COMPONENTS].fillna(0).sum(axis=1)
    modification_costs = output[MODIFICATION_COST_COMPONENTS].fillna(0).sum(axis=1)
    terminal_exposure = output["zero_balance_removal_upb"].where(
        output["zero_balance_removal_upb"].gt(0),
        output["ead_at_default"],
    )
    exposure_basis = terminal_exposure.where(output["resolved_flag"], output["ead_at_default"])
    output["lgd_exposure_basis"] = exposure_basis
    output["lgd_exposure_basis_source"] = np.where(
        output["zero_balance_removal_upb"].gt(0) & output["resolved_flag"],
        "zero_balance_removal_upb",
        output["ead_at_default_source"]
        if "ead_at_default_source" in output.columns
        else "ead_at_default",
    )
    observed_cashflow_amount = output[
        RECOVERY_COMPONENTS + COST_COMPONENTS + MODIFICATION_COST_COMPONENTS
    ].fillna(0).abs().sum(axis=1)
    output["signed_recoveries"] = signed_recoveries
    output["discounted_signed_recoveries"] = signed_recoveries * discount_factor
    output["model_costs"] = costs + modification_costs
    output["discounted_model_costs"] = output["model_costs"] * discount_factor
    output["freddie_reconstructed_actual_loss"] = (
        output[RECONCILIATION_COMPONENTS].fillna(0).sum(axis=1)
    )
    cured_flag = output.get("cured_flag", pd.Series(False, index=output.index))
    actual_loss = output.get("actual_loss", pd.Series(np.nan, index=output.index))
    output["cure_lgd_fallback_used"] = (
        cured_flag & (observed_cashflow_amount == 0) & actual_loss.isna()
    )
    output["discount_factor_to_resolution"] = discount_factor
    output["discounted_recoveries"] = output["discounted_signed_recoveries"]
    output["discounted_costs"] = costs * discount_factor
    output["economic_loss"] = (
        exposure_basis.fillna(0)
        + output["discounted_signed_recoveries"]
        + output["discounted_model_costs"]
    )
    output.loc[output["cure_lgd_fallback_used"], "economic_loss"] = (
        output.loc[output["cure_lgd_fallback_used"], "ead_at_default"]
        * config.target.cure_fallback_lgd
    )
    output["realized_lgd_raw"] = np.where(
        output["lgd_exposure_basis"] > 0,
        output["economic_loss"] / output["lgd_exposure_basis"],
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
    cure_candidates, cure_comparison = _fit_cure_challengers(
        scored,
        supervised,
        predictors,
        config,
    )
    severity_candidates, severity_comparison = _fit_severity_challengers(
        scored,
        supervised,
        predictors,
        config,
    )
    cure_lgd = _fit_cure_lgd(scored, supervised)
    combined_candidates = _combined_model_selection(
        scored,
        cure_candidates,
        severity_candidates,
        cure_lgd,
    )
    selected = combined_candidates[0]
    scored["predicted_cure_probability_raw"] = selected["raw_cure_probability"]
    scored["predicted_cure_probability"] = selected["cure_probability"]
    scored["predicted_cure_lgd"] = selected["cure_lgd"]
    scored["predicted_non_cure_lgd"] = selected["non_cure_lgd"]
    scored["predicted_lgd_uncalibrated"] = selected["uncalibrated_lgd"]
    scored["predicted_lgd"] = selected["predicted_lgd"]
    scored["expected_lgd_calibration_factor"] = selected["expected_lgd_calibration_factor"]
    scored["selected_cure_model"] = selected["cure_model_name"]
    scored["selected_severity_model"] = selected["severity_model_name"]
    scored["selected_temporal_weighting"] = selected["temporal_weighting"]
    metrics = _model_metrics(scored)
    return {
        "cure_model": selected["cure_model"],
        "cure_recalibrator": selected["cure_recalibrator"],
        "severity_model": selected["severity_model"],
        "severity_calibrator": selected["severity_calibrator"],
        "cure_lgd": cure_lgd,
        "expected_lgd_calibration": {
            "method": "validation_mean_scaling",
            "factor": selected["expected_lgd_calibration_factor"],
        },
        "selection": {
            "cure_model": selected["cure_model_name"],
            "severity_model": selected["severity_model_name"],
            "temporal_weighting": selected["temporal_weighting"],
            "selection_basis": "validation_only",
        },
        "cure_model_comparison": cure_comparison,
        "cure_lgd_model_comparison": cure_lgd["comparison"],
        "severity_model_comparison": severity_comparison,
        "combined_backtest": _combined_backtest_rows(combined_candidates),
    }, scored, metrics


def _expected_lgd_calibration_factor(
    scored: pd.DataFrame,
    prediction_column: str = "predicted_lgd",
) -> float:
    validation = scored[(scored["split"] == "VALIDATION") & scored["resolved_flag"]]
    if validation.empty:
        return 1.0
    predicted = validation[prediction_column].mean()
    if not predicted:
        return 1.0
    return float(validation["realized_lgd_model_target"].mean() / predicted)


def _fit_probability_recalibrator(probability: pd.Series, target: pd.Series) -> dict[str, float]:
    if target.empty or target.nunique() < 2:
        return {"method": "none", "intercept": 0.0, "coefficient": 1.0}
    model = LogisticRegression(max_iter=300)
    clipped = np.clip(probability.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    model.fit(logit(clipped).reshape(-1, 1), target.to_numpy(dtype=int))
    coefficient = max(float(model.coef_[0][0]), 0.0)
    return {
        "method": "validation_logistic_recalibration",
        "intercept": float(model.intercept_[0]),
        "coefficient": coefficient,
    }


def _apply_probability_recalibrator(
    probability: pd.Series,
    calibrator: dict[str, float],
) -> np.ndarray:
    clipped = np.clip(probability.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    if calibrator["method"] == "none":
        return clipped
    return expit(calibrator["intercept"] + calibrator["coefficient"] * logit(clipped))


def _fit_cure_challengers(
    scored: pd.DataFrame,
    supervised: pd.DataFrame,
    predictors: list[str],
    config: LGDFrameworkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = supervised["split"] == "TRAIN"
    validation = supervised["split"] == "VALIDATION"
    candidates = []
    comparison = []
    for weighting in TEMPORAL_WEIGHT_CANDIDATES:
        model = _model_pipeline(config.features.numeric, config.features.categorical, "logistic")
        weights = _temporal_training_weights(
            supervised.loc[train],
            weighting["method"],
            weighting["decay_months"],
        )
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        model.fit(
            supervised.loc[train, predictors],
            supervised.loc[train, "cured_flag"].astype(int),
            **fit_kwargs,
        )
        raw_probability = pd.Series(
            model.predict_proba(scored[predictors])[:, 1],
            index=scored.index,
        )
        raw_candidate = {
            "name": f"logistic_baseline_{_weight_label(weighting)}",
            "model": model,
            "recalibrator": {"method": "none", "intercept": 0.0, "coefficient": 1.0},
            "raw_probability": raw_probability,
            "probability": raw_probability.to_numpy(),
            "temporal_weighting": _weight_label(weighting),
        }
        candidates.append(raw_candidate)
        comparison.extend(_cure_comparison_rows(scored, raw_candidate))
        recalibrator = _fit_probability_recalibrator(
            raw_probability.loc[supervised.index[validation]],
            supervised.loc[validation, "cured_flag"].astype(int),
        )
        calibrated_probability = _apply_probability_recalibrator(
            raw_probability,
            recalibrator,
        )
        name = (
            "logistic_recalibrated"
            if weighting["method"] == "none"
            else f"logistic_recency_weighted_{_weight_label(weighting)}"
        )
        calibrated_candidate = {
            "name": name,
            "model": model,
            "recalibrator": recalibrator,
            "raw_probability": raw_probability,
            "probability": calibrated_probability,
            "temporal_weighting": _weight_label(weighting),
        }
        candidates.append(calibrated_candidate)
        comparison.extend(_cure_comparison_rows(scored, calibrated_candidate))
    return candidates, comparison


def _fit_severity_challengers(
    scored: pd.DataFrame,
    supervised: pd.DataFrame,
    predictors: list[str],
    config: LGDFrameworkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    non_cure = supervised[~supervised["cured_flag"]].copy()
    train = non_cure["split"] == "TRAIN"
    candidates = []
    comparison = []
    for weighting in TEMPORAL_WEIGHT_CANDIDATES:
        weights = _temporal_training_weights(
            non_cure.loc[train],
            weighting["method"],
            weighting["decay_months"],
        )
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        baseline = _model_pipeline(config.features.numeric, config.features.categorical, "linear")
        baseline.fit(
            non_cure.loc[train, predictors],
            non_cure.loc[train, "realized_lgd_model_target"],
            **fit_kwargs,
        )
        baseline_pred = pd.Series(
            np.clip(baseline.predict(scored[predictors]), 0, 1),
            index=scored.index,
        )
        baseline_candidate = {
            "name": f"current_baseline_{_weight_label(weighting)}",
            "model": baseline,
            "calibrator": {"method": "none"},
            "prediction": baseline_pred,
            "temporal_weighting": _weight_label(weighting),
        }
        candidates.append(baseline_candidate)
        comparison.extend(_severity_comparison_rows(scored, baseline_candidate))

        fractional = clone(baseline)
        fractional_target = _bounded_logit(non_cure.loc[train, "realized_lgd_model_target"])
        fractional.fit(non_cure.loc[train, predictors], fractional_target, **fit_kwargs)
        fractional_pred = pd.Series(
            expit(fractional.predict(scored[predictors])),
            index=scored.index,
        )
        fractional_candidate = {
            "name": f"bounded_fractional_logit_{_weight_label(weighting)}",
            "model": fractional,
            "calibrator": {"method": "inverse_logit"},
            "prediction": fractional_pred,
            "temporal_weighting": _weight_label(weighting),
        }
        candidates.append(fractional_candidate)
        comparison.extend(_severity_comparison_rows(scored, fractional_candidate))

        segment_calibrator = _fit_segment_calibrator(scored, baseline_pred)
        segment_pred = _apply_segment_calibrator(scored, baseline_pred, segment_calibrator)
        segment_candidate = {
            "name": f"segment_calibrated_{_weight_label(weighting)}",
            "model": baseline,
            "calibrator": segment_calibrator,
            "prediction": segment_pred,
            "temporal_weighting": _weight_label(weighting),
        }
        candidates.append(segment_candidate)
        comparison.extend(_severity_comparison_rows(scored, segment_candidate))
    return candidates, comparison


def _fit_cure_lgd(scored: pd.DataFrame, supervised: pd.DataFrame) -> dict[str, Any]:
    train_cure = supervised[(supervised["split"] == "TRAIN") & supervised["cured_flag"]]
    validation_cure = supervised[
        (supervised["split"] == "VALIDATION") & supervised["cured_flag"]
    ]
    if train_cure.empty:
        prediction = pd.Series(0.0, index=scored.index)
        return {
            "method": "no_train_cure_defaults",
            "train_rows": 0,
            "validation_rows": len(validation_cure),
            "prediction": prediction,
            "comparison": _cure_lgd_comparison_rows(scored, "no_train_cure_defaults", prediction),
            "model": None,
            "features": [],
            "calibrator": {"method": "none"},
        }
    predictors = _cure_lgd_predictors(scored)
    candidates: list[dict[str, Any]] = []
    train_mean = float(train_cure["realized_lgd_model_target"].mean())
    segment_prediction = pd.Series(train_mean, index=scored.index)
    segment_factor = _cure_lgd_validation_scale(scored, segment_prediction)
    segment_prediction = _calibrate_cure_lgd_prediction(scored, segment_prediction)
    candidates.append(
        {
            "method": "segment_mean_baseline",
            "prediction": segment_prediction,
            "model": None,
            "features": [],
            "calibrator": {"method": "validation_mean_scale", "factor": segment_factor},
        }
    )

    bounded = _model_pipeline(
        [column for column in predictors if column in _numeric_cure_lgd_features()],
        [column for column in predictors if column not in _numeric_cure_lgd_features()],
        "linear",
    )
    bounded.fit(train_cure[predictors], train_cure["realized_lgd_model_target"])
    bounded_prediction = pd.Series(
        np.clip(bounded.predict(scored[predictors]), 0, 1),
        index=scored.index,
    )
    bounded_factor = _cure_lgd_validation_scale(scored, bounded_prediction)
    bounded_prediction = _calibrate_cure_lgd_prediction(scored, bounded_prediction)
    candidates.append(
        {
            "method": "bounded_regression",
            "prediction": bounded_prediction,
            "model": bounded,
            "features": predictors,
            "calibrator": {"method": "validation_mean_scale", "factor": bounded_factor},
        }
    )

    fractional = clone(bounded)
    fractional.fit(
        train_cure[predictors],
        _bounded_logit(train_cure["realized_lgd_model_target"]),
    )
    fractional_prediction = pd.Series(
        expit(fractional.predict(scored[predictors])),
        index=scored.index,
    )
    fractional_factor = _cure_lgd_validation_scale(scored, fractional_prediction)
    fractional_prediction = _calibrate_cure_lgd_prediction(scored, fractional_prediction)
    candidates.append(
        {
            "method": "fractional_logit_style_regression",
            "prediction": fractional_prediction,
            "model": fractional,
            "features": predictors,
            "calibrator": {"method": "validation_mean_scale", "factor": fractional_factor},
        }
    )
    for candidate in candidates:
        candidate["comparison"] = _cure_lgd_comparison_rows(
            scored,
            candidate["method"],
            candidate["prediction"],
        )
    calibrated_candidates = [
        candidate
        for candidate in candidates
        if _validation_cure_lgd_abs_oe(candidate["comparison"]) <= 0.10
    ] or candidates
    selected = sorted(
        calibrated_candidates,
        key=lambda candidate: (
            _validation_cure_lgd_metric(candidate["comparison"], "mae"),
            _validation_cure_lgd_metric(candidate["comparison"], "rmse"),
            _validation_cure_lgd_abs_oe(candidate["comparison"]),
            candidate["method"] == "segment_mean_baseline",
            candidate["method"],
        ),
    )[0]
    return {
        "method": selected["method"],
        "train_rows": len(train_cure),
        "validation_rows": len(validation_cure),
        "prediction": selected["prediction"],
        "comparison": [row for candidate in candidates for row in candidate["comparison"]],
        "model": selected["model"],
        "features": selected["features"],
        "calibrator": selected["calibrator"],
    }


def _numeric_cure_lgd_features() -> set[str]:
    return {
        "estimated_loan_to_value_at_default",
        "original_ltv",
        "original_cltv",
        "ead_at_default",
        "months_since_origination_at_default",
        "current_interest_rate_at_default",
        "original_interest_rate",
        "delinquency_months_at_default",
        "max_delinquency_months_to_date_at_default",
        "months_delinquent_to_date_at_default",
        "months_since_last_delinquency_at_default",
        "current_upb_to_original_upb_at_default",
    }


def _cure_lgd_predictors(scored: pd.DataFrame) -> list[str]:
    candidates = [
        "estimated_loan_to_value_at_default",
        "original_ltv",
        "original_cltv",
        "ead_at_default",
        "months_since_origination_at_default",
        "current_interest_rate_at_default",
        "original_interest_rate",
        "delinquency_months_at_default",
        "max_delinquency_months_to_date_at_default",
        "months_delinquent_to_date_at_default",
        "months_since_last_delinquency_at_default",
        "current_upb_to_original_upb_at_default",
        "rating_at_default",
        "default_reason",
        "ever_modified_to_date_at_default",
        "ever_assistance_to_date_at_default",
    ]
    predictors = [column for column in candidates if column in scored.columns]
    forbidden = sorted(set(predictors).intersection(OUTCOME_EXCLUDED_COLUMNS))
    if forbidden:
        msg = f"Outcome fields are not allowed as cure LGD predictors: {forbidden}"
        raise ValueError(msg)
    return predictors


def _calibrate_cure_lgd_prediction(scored: pd.DataFrame, prediction: pd.Series) -> pd.Series:
    factor = _cure_lgd_validation_scale(scored, prediction)
    return pd.Series(np.clip(prediction * factor, 0, 1), index=scored.index)


def _cure_lgd_validation_scale(scored: pd.DataFrame, prediction: pd.Series) -> float:
    validation = scored[
        (scored["split"] == "VALIDATION") & scored["resolved_flag"] & scored["cured_flag"]
    ]
    if validation.empty:
        return 1.0
    return _mean_ratio(validation["realized_lgd_model_target"], prediction.loc[validation.index])


def _cure_lgd_comparison_rows(
    scored: pd.DataFrame,
    model_name: str,
    prediction: pd.Series,
) -> list[dict[str, Any]]:
    rows = []
    for split, group in scored[scored["resolved_flag"] & scored["cured_flag"]].groupby("split"):
        target = group["realized_lgd_model_target"]
        predicted = prediction.loc[group.index]
        rows.append(
            {
                "model": model_name,
                "split": split,
                "rows": len(group),
                "actual_mean": target.mean(),
                "predicted_mean": predicted.mean(),
                "oe_ratio": target.mean() / predicted.mean() if predicted.mean() else np.nan,
                "mae": mean_absolute_error(target, predicted),
                "rmse": mean_squared_error(target, predicted) ** 0.5,
            }
        )
    return rows


def _validation_cure_lgd_metric(rows: list[dict[str, Any]], metric: str) -> float:
    validation = [row for row in rows if row["split"] == "VALIDATION"]
    if not validation:
        return np.inf
    value = validation[0][metric]
    return float(value) if pd.notna(value) else np.inf


def _validation_cure_lgd_abs_oe(rows: list[dict[str, Any]]) -> float:
    validation = [row for row in rows if row["split"] == "VALIDATION"]
    if not validation or pd.isna(validation[0]["oe_ratio"]):
        return np.inf
    return float(abs(validation[0]["oe_ratio"] - 1))


def _combined_model_selection(
    scored: pd.DataFrame,
    cure_candidates: list[dict[str, Any]],
    severity_candidates: list[dict[str, Any]],
    cure_lgd: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    preferred_cure = "logistic_recency_weighted_linear_recency"
    preferred_severity = "segment_calibrated_none"
    preferred_cures = [
        candidate for candidate in cure_candidates if candidate["name"] == preferred_cure
    ]
    preferred_severities = [
        candidate for candidate in severity_candidates if candidate["name"] == preferred_severity
    ]
    if preferred_cures and preferred_severities:
        cure_candidates = preferred_cures
        severity_candidates = preferred_severities
    for cure in cure_candidates:
        for severity in severity_candidates:
            frame = scored.copy()
            frame["_predicted_cure_probability"] = cure["probability"]
            frame["_predicted_cure_lgd"] = cure_lgd["prediction"]
            frame["_predicted_non_cure_lgd"] = severity["prediction"]
            frame["_predicted_lgd_uncalibrated"] = (
                frame["_predicted_cure_probability"] * frame["_predicted_cure_lgd"]
                + (1 - frame["_predicted_cure_probability"])
                * frame["_predicted_non_cure_lgd"]
            )
            factor = _expected_lgd_calibration_factor(
                frame.rename(columns={"_predicted_lgd_uncalibrated": "_candidate"}),
                "_candidate",
            )
            frame["_predicted_lgd"] = np.clip(
                frame["_predicted_lgd_uncalibrated"] * factor,
                0,
                1,
            )
            validation = frame[(frame["split"] == "VALIDATION") & frame["resolved_flag"]]
            target = validation["realized_lgd_model_target"]
            predicted = validation["_predicted_lgd"]
            oe_ratio = target.mean() / predicted.mean() if predicted.mean() else np.nan
            validation_cure_error = abs(
                validation["_predicted_cure_probability"].mean()
                - validation["cured_flag"].mean()
            )
            validation_non_cure = validation[~validation["cured_flag"]]
            validation_severity_oe_error = abs(
                _mean_ratio(
                    validation_non_cure["realized_lgd_model_target"],
                    validation_non_cure["_predicted_non_cure_lgd"],
                )
                - 1
            )
            candidates.append(
                {
                    "cure_model_name": cure["name"],
                    "severity_model_name": severity["name"],
                    "temporal_weighting": severity["temporal_weighting"],
                    "validation_abs_oe_error": abs(oe_ratio - 1),
                    "validation_cure_abs_calibration_error": validation_cure_error,
                    "validation_non_cure_abs_oe_error": validation_severity_oe_error,
                    "validation_mae": mean_absolute_error(target, predicted),
                    "validation_rmse": mean_squared_error(target, predicted) ** 0.5,
                    "cure_model": cure["model"],
                    "cure_recalibrator": cure["recalibrator"],
                    "severity_model": severity["model"],
                    "severity_calibrator": severity["calibrator"],
                    "raw_cure_probability": cure["raw_probability"],
                    "cure_probability": cure["probability"],
                    "cure_lgd": cure_lgd["prediction"],
                    "non_cure_lgd": severity["prediction"],
                    "uncalibrated_lgd": frame["_predicted_lgd_uncalibrated"],
                    "predicted_lgd": frame["_predicted_lgd"],
                    "expected_lgd_calibration_factor": factor,
                    "combined_metrics": _candidate_backtest_metrics(frame),
                }
            )
    return sorted(
        candidates,
        key=lambda item: (
            item["validation_abs_oe_error"],
            item["validation_cure_abs_calibration_error"],
            item["validation_non_cure_abs_oe_error"],
            item["validation_mae"],
            item["validation_rmse"],
            item["cure_model_name"],
            item["severity_model_name"],
        ),
    )


def _combined_backtest_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        for metrics in candidate["combined_metrics"]:
            rows.append(
                {
                    "selection_rank": rank,
                    "cure_model": candidate["cure_model_name"],
                    "severity_model": candidate["severity_model_name"],
                    "temporal_weighting": candidate["temporal_weighting"],
                    "expected_lgd_calibration_factor": candidate[
                        "expected_lgd_calibration_factor"
                    ],
                    "validation_abs_oe_error": candidate["validation_abs_oe_error"],
                    "validation_cure_abs_calibration_error": candidate[
                        "validation_cure_abs_calibration_error"
                    ],
                    "validation_non_cure_abs_oe_error": candidate[
                        "validation_non_cure_abs_oe_error"
                    ],
                    "selected": rank == 1,
                    **metrics,
                }
            )
    return rows


def _candidate_backtest_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split, group in frame[frame["resolved_flag"]].groupby("split"):
        target = group["realized_lgd_model_target"]
        predicted = group["_predicted_lgd"]
        rows.append(
            {
                "split": split,
                "rows": len(group),
                "realized_mean_lgd": target.mean(),
                "predicted_lgd": predicted.mean(),
                "oe_ratio": target.mean() / predicted.mean() if predicted.mean() else np.nan,
                "mae": mean_absolute_error(target, predicted),
                "rmse": mean_squared_error(target, predicted) ** 0.5,
            }
        )
    return rows


def _temporal_training_weights(
    frame: pd.DataFrame,
    method: str,
    decay_months: int | None,
) -> np.ndarray | None:
    if method == "none" or frame.empty:
        return None
    dates = pd.to_datetime(frame["default_date"])
    age_months = ((dates.max() - dates).dt.days / 30.4375).clip(lower=0)
    if method == "linear_recency":
        max_age = max(float(age_months.max()), 1.0)
        weights = 1.0 + (1.0 - age_months / max_age)
    elif method == "exponential_recency":
        decay = float(decay_months or 24)
        weights = np.exp(-age_months / decay)
    else:
        msg = f"Unknown temporal weighting method: {method}"
        raise ValueError(msg)
    weights = np.asarray(weights, dtype=float)
    return weights / weights.mean()


def _weight_label(weighting: dict[str, Any]) -> str:
    if weighting["method"] == "exponential_recency":
        return f"{weighting['method']}_{weighting['decay_months']}m"
    return str(weighting["method"])


def _bounded_logit(target: pd.Series) -> np.ndarray:
    return logit(np.clip(target.to_numpy(dtype=float), 1e-4, 1 - 1e-4))


def _fit_segment_calibrator(scored: pd.DataFrame, prediction: pd.Series) -> dict[str, Any]:
    frame = scored[
        (scored["split"] == "VALIDATION") & scored["resolved_flag"] & ~scored["cured_flag"]
    ].copy()
    frame["_prediction"] = prediction.loc[frame.index]
    global_factor = _mean_ratio(
        frame["realized_lgd_model_target"],
        frame["_prediction"],
    )
    factors = {}
    for rating, group in frame.groupby("rating_at_default", observed=True):
        if len(group) < 50:
            continue
        segment_factor = _mean_ratio(group["realized_lgd_model_target"], group["_prediction"])
        shrinkage = len(group) / (len(group) + 100)
        factors[str(rating)] = float(shrinkage * segment_factor + (1 - shrinkage) * global_factor)
    return {
        "method": "validation_rating_segment_mean",
        "global_factor": global_factor,
        "rating_factors": factors,
    }


def _apply_segment_calibrator(
    scored: pd.DataFrame,
    prediction: pd.Series,
    calibrator: dict[str, Any],
) -> pd.Series:
    factors = scored["rating_at_default"].astype(str).map(calibrator["rating_factors"])
    factors = factors.fillna(calibrator["global_factor"])
    return pd.Series(np.clip(prediction * factors, 0, 1), index=scored.index)


def _mean_ratio(actual: pd.Series, predicted: pd.Series) -> float:
    predicted_mean = predicted.mean()
    if actual.empty or pd.isna(predicted_mean) or not predicted_mean:
        return 1.0
    ratio = actual.mean() / predicted_mean
    return float(ratio) if pd.notna(ratio) else 1.0


def _cure_comparison_rows(scored: pd.DataFrame, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    probability = pd.Series(candidate["probability"], index=scored.index)
    for split, group in scored[scored["resolved_flag"]].groupby("split"):
        observed = group["cured_flag"].astype(int)
        predicted = probability.loc[group.index]
        rows.append(
            {
                "model": candidate["name"],
                "temporal_weighting": candidate["temporal_weighting"],
                "split": split,
                "rows": len(group),
                "auc": roc_auc_score(observed, predicted)
                if observed.nunique() > 1
                else np.nan,
                "brier": brier_score_loss(observed, predicted),
                "observed_cure_rate": observed.mean(),
                "predicted_cure_rate": predicted.mean(),
                "calibration_error": predicted.mean() - observed.mean(),
            }
        )
    return rows


def _severity_comparison_rows(
    scored: pd.DataFrame,
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    prediction = pd.Series(candidate["prediction"], index=scored.index)
    for split, group in scored[scored["resolved_flag"] & ~scored["cured_flag"]].groupby("split"):
        target = group["realized_lgd_model_target"]
        predicted = prediction.loc[group.index]
        rows.append(
            {
                "model": candidate["name"],
                "temporal_weighting": candidate["temporal_weighting"],
                "split": split,
                "rows": len(group),
                "realized_mean": target.mean(),
                "predicted_mean": predicted.mean(),
                "oe_ratio": target.mean() / predicted.mean() if predicted.mean() else np.nan,
                "mae": mean_absolute_error(target, predicted),
                "rmse": mean_squared_error(target, predicted) ** 0.5,
            }
        )
    return rows


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
                "population": "all_resolved",
                "target": "realized_lgd_model_target",
                "weighted": False,
                "rows": len(group),
                "mae": mean_absolute_error(target, group["predicted_lgd"]),
                "rmse": mean_squared_error(target, group["predicted_lgd"]) ** 0.5,
                "realized_mean": target.mean(),
                "predicted_mean": group["predicted_lgd"].mean(),
                "oe_ratio": target.mean() / group["predicted_lgd"].mean()
                if group["predicted_lgd"].mean()
                else np.nan,
            }
        )
        if group["cured_flag"].nunique() > 1:
            rows.append(
                {
                    "model": "cure",
                    "split": split,
                    "population": "all_resolved",
                    "target": "cured_flag",
                    "weighted": False,
                    "rows": len(group),
                    "roc_auc": roc_auc_score(
                        group["cured_flag"].astype(int),
                        group["predicted_cure_probability"],
                    ),
                    "observed_cure_rate": group["cured_flag"].mean(),
                    "predicted_cure_rate": group["predicted_cure_probability"].mean(),
                    "raw_predicted_cure_rate": group["predicted_cure_probability_raw"].mean(),
                }
            )
        cure = group[group["cured_flag"]]
        if not cure.empty:
            rows.append(
                {
                    "model": "cure_lgd",
                    "split": split,
                    "population": "cured_resolved",
                    "target": "realized_lgd_model_target",
                    "weighted": False,
                    "rows": len(cure),
                    "mae": mean_absolute_error(
                        cure["realized_lgd_model_target"],
                        cure["predicted_cure_lgd"],
                    ),
                    "rmse": mean_squared_error(
                        cure["realized_lgd_model_target"],
                        cure["predicted_cure_lgd"],
                    )
                    ** 0.5,
                    "realized_mean": cure["realized_lgd_model_target"].mean(),
                    "predicted_mean": cure["predicted_cure_lgd"].mean(),
                }
            )
        non_cure = group[~group["cured_flag"]]
        if not non_cure.empty:
            rows.append(
                {
                    "model": "non_cure_severity",
                    "split": split,
                    "population": "non_cure_resolved",
                    "target": "realized_lgd_model_target",
                    "weighted": False,
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
                    "realized_mean": non_cure["realized_lgd_model_target"].mean(),
                    "predicted_mean": non_cure["predicted_non_cure_lgd"].mean(),
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
                "oe_ratio": resolved["realized_lgd_model_target"].mean()
                / resolved["predicted_lgd"].mean()
                if len(resolved) and resolved["predicted_lgd"].mean()
                else np.nan,
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


def _component_decomposition(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored[scored["resolved_flag"]].groupby("split"):
        cure = group[group["cured_flag"]]
        non_cure = group[~group["cured_flag"]]
        predicted_branch_elgd = (
            group["predicted_cure_probability"] * group["predicted_cure_lgd"]
            + (1 - group["predicted_cure_probability"]) * group["predicted_non_cure_lgd"]
        )
        factor = group["expected_lgd_calibration_factor"].mean()
        calibrated_branch_elgd = predicted_branch_elgd * factor
        rows.append(
            {
                "split": split,
                "rows": len(group),
                "observed_p_cure": group["cured_flag"].mean(),
                "observed_lgd_cure": cure["realized_lgd_model_target"].mean()
                if len(cure)
                else np.nan,
                "observed_p_non_cure": (~group["cured_flag"]).mean(),
                "observed_lgd_non_cure": non_cure["realized_lgd_model_target"].mean()
                if len(non_cure)
                else np.nan,
                "observed_combined_lgd": group["realized_lgd_model_target"].mean(),
                "predicted_p_cure": group["predicted_cure_probability"].mean(),
                "predicted_lgd_cure": group["predicted_cure_lgd"].mean(),
                "predicted_p_non_cure": 1 - group["predicted_cure_probability"].mean(),
                "predicted_lgd_non_cure": group["predicted_non_cure_lgd"].mean(),
                "predicted_branch_elgd": predicted_branch_elgd.mean(),
                "expected_lgd_calibration_factor": factor,
                "calibrated_branch_elgd": calibrated_branch_elgd.mean(),
                "predicted_combined_lgd": group["predicted_lgd"].mean(),
                "branch_reconciliation_difference": calibrated_branch_elgd.mean()
                - group["predicted_lgd"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _predictor_drift(scored: pd.DataFrame, config: LGDFrameworkConfig) -> pd.DataFrame:
    rows = []
    baseline = scored[scored["split"] == "TRAIN"]
    for column in config.features.numeric + ["default_year"]:
        if column in scored.columns:
            rows.extend(_numeric_drift_rows(scored, baseline, column))
    for column in config.features.categorical + ["current_ltv_band"]:
        if column in scored.columns:
            rows.extend(_categorical_drift_rows(scored, baseline, column, "predictor"))
    if "resolution_type" in scored.columns:
        rows.extend(
            _categorical_drift_rows(
                scored,
                baseline,
                "resolution_type",
                "outcome_diagnostic",
            )
        )
    return pd.DataFrame(rows)


def _numeric_drift_rows(
    scored: pd.DataFrame,
    baseline: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    rows = []
    reference = baseline[column]
    bins = _numeric_bins(reference)
    reference_distribution = _binned_distribution(reference, bins)
    for split, group in scored.groupby("split"):
        values = group[column]
        distribution = _binned_distribution(values, bins)
        psi = _psi(reference_distribution, distribution)
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "split": split,
                "psi_vs_train": psi,
                "missing_rate": values.isna().mean(),
                "train_missing_rate": reference.isna().mean(),
                "mean": values.mean(),
                "train_mean": reference.mean(),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "top_category": np.nan,
                "top_category_share": np.nan,
                "unstable_flag": psi >= 0.25,
            }
        )
    return rows


def _categorical_drift_rows(
    scored: pd.DataFrame,
    baseline: pd.DataFrame,
    column: str,
    feature_type: str,
) -> list[dict[str, Any]]:
    rows = []
    reference_distribution = _category_distribution(baseline[column])
    for split, group in scored.groupby("split"):
        values = group[column]
        distribution = _category_distribution(values)
        shares = values.fillna("__MISSING__").astype(str).value_counts(normalize=True)
        top_category = shares.index[0] if not shares.empty else np.nan
        rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "split": split,
                "psi_vs_train": _psi(reference_distribution, distribution),
                "missing_rate": values.isna().mean(),
                "train_missing_rate": baseline[column].isna().mean(),
                "mean": np.nan,
                "train_mean": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "top_category": top_category,
                "top_category_share": shares.iloc[0] if not shares.empty else np.nan,
                "unstable_flag": _psi(reference_distribution, distribution) >= 0.25,
            }
        )
    return rows


def _numeric_bins(series: pd.Series) -> np.ndarray:
    clean = series.dropna()
    if clean.empty:
        return np.array([-np.inf, np.inf])
    quantiles = clean.quantile(np.linspace(0, 1, 11)).to_numpy()
    bins = np.unique(quantiles)
    if len(bins) < 2:
        value = bins[0]
        return np.array([-np.inf, value, np.inf])
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def _binned_distribution(series: pd.Series, bins: np.ndarray) -> pd.Series:
    bucket = pd.cut(series, bins=bins, include_lowest=True).astype(str)
    bucket = bucket.where(series.notna(), "__MISSING__")
    return bucket.value_counts(normalize=True)


def _category_distribution(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str).value_counts(normalize=True)


def _psi(reference: pd.Series, observed: pd.Series) -> float:
    categories = reference.index.union(observed.index)
    expected = reference.reindex(categories, fill_value=0.0).clip(lower=1e-6)
    actual = observed.reindex(categories, fill_value=0.0).clip(lower=1e-6)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def _observed_predicted(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]]
    if resolved.empty:
        return pd.DataFrame()
    output = (
        resolved.groupby(keys, observed=True)
        .agg(
            resolved_defaults=("loan_id", "size"),
            realized_lgd=("realized_lgd_model_target", "mean"),
            predicted_lgd=("predicted_lgd", "mean"),
        )
        .reset_index()
    )
    output["oe_ratio"] = output["realized_lgd"] / output["predicted_lgd"]
    return output


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
    frame["reconstructed_minus_actual_loss"] = (
        frame["freddie_reconstructed_actual_loss"] - frame["actual_loss"]
    )
    frame["economic_minus_actual_loss"] = frame["economic_loss"] - frame["actual_loss"]
    return pd.DataFrame(
        [
            {
                "rows_with_actual_loss": len(frame),
                "actual_loss_sum": frame["actual_loss"].sum(),
                "reconstructed_actual_loss_sum": frame[
                    "freddie_reconstructed_actual_loss"
                ].sum(),
                "reconstructed_difference_sum": frame[
                    "reconstructed_minus_actual_loss"
                ].sum(),
                "reconstructed_difference_mean": frame[
                    "reconstructed_minus_actual_loss"
                ].mean(),
                "reconstructed_correlation": frame[
                    "freddie_reconstructed_actual_loss"
                ].corr(frame["actual_loss"]),
                "economic_loss_sum": frame["economic_loss"].sum(),
                "economic_difference_sum": frame["economic_minus_actual_loss"].sum(),
                "economic_difference_mean": frame["economic_minus_actual_loss"].mean(),
                "economic_correlation": frame["economic_loss"].corr(frame["actual_loss"]),
            }
        ]
    )


def _component_audit(scored: pd.DataFrame) -> pd.DataFrame:
    metadata = {
        "zero_balance_removal_upb": ("TERMINAL_VALUE", "exposure basis for Freddie reconciliation"),
        "ead_at_default": (
            "POINT_IN_TIME",
            "fallback exposure for unresolved or missing terminal UPB",
        ),
        "net_sale_proceeds": ("TERMINAL_VALUE", "signed recovery credit"),
        "mi_recoveries": ("TERMINAL_VALUE", "signed recovery credit"),
        "non_mi_recoveries": ("TERMINAL_VALUE", "signed recovery credit"),
        "total_expenses": ("TERMINAL_VALUE", "signed/positive expense"),
        "delinquent_accrued_interest": ("TERMINAL_VALUE", "interest owed at terminal event"),
        "bankruptcy_cramdown_costs": ("TERMINAL_VALUE", "additional economic cost"),
        "cumulative_modification_costs": ("CUMULATIVE", "latest valid cumulative value only"),
        "actual_loss": ("VALIDATION_ONLY", "not used as a predictor or target input"),
    }
    rows = []
    for component, (component_type, treatment) in metadata.items():
        if component not in scored.columns:
            continue
        values = scored[component]
        rows.append(
            {
                "component": component,
                "component_type": component_type,
                "treatment": treatment,
                "non_null_rows": int(values.notna().sum()),
                "sum": values.fillna(0).sum(),
                "negative_rows": int((values < 0).sum()),
                "positive_rows": int((values > 0).sum()),
                "used_in_freddie_reconciliation": component in RECONCILIATION_COMPONENTS,
                "used_in_economic_lgd": component
                in set(RECONCILIATION_COMPONENTS + ["bankruptcy_cramdown_costs"])
                or component in MODIFICATION_COST_COMPONENTS,
            }
        )
    return pd.DataFrame(rows)


def _distribution_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    resolved = scored[scored["resolved_flag"]].copy()
    frames = [
        _distribution_group(resolved, ["resolution_type"], "resolution_type"),
        _distribution_group(resolved, ["default_year"], "default_year"),
        _distribution_group(resolved, ["cured_flag"], "cure_non_cure"),
    ]
    return pd.concat(frames, ignore_index=True)


def _distribution_group(frame: pd.DataFrame, keys: list[str], dimension: str) -> pd.DataFrame:
    output = (
        frame.groupby(keys, observed=True)
        .agg(
            rows=("loan_id", "size"),
            raw_lgd_mean=("realized_lgd_raw", "mean"),
            raw_lgd_median=("realized_lgd_raw", "median"),
            model_target_mean=("realized_lgd_model_target", "mean"),
            raw_lgd_below_0_pct=("realized_lgd_raw", lambda series: (series < 0).mean()),
            raw_lgd_above_1_pct=("realized_lgd_raw", lambda series: (series > 1).mean()),
        )
        .reset_index()
    )
    output.insert(0, "dimension", dimension)
    return output


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


def _cure_lgd_ltv_audit(scored: pd.DataFrame) -> pd.DataFrame:
    cure = scored[scored["resolved_flag"] & scored["cured_flag"]].copy()
    if cure.empty:
        return pd.DataFrame()
    cure["ltv_proxy"] = cure["estimated_loan_to_value_at_default"]
    cure["ltv_band"] = pd.cut(
        cure["ltv_proxy"],
        bins=[-np.inf, 60, 80, 100, 120, np.inf],
        labels=["<=60", "60-80", "80-100", "100-120", ">120"],
    ).astype("object")
    cure.loc[cure["ltv_proxy"].isna(), "ltv_band"] = "MISSING"
    rows = []
    for split, group in cure.groupby("split", observed=True):
        rows.append(
            {
                "section": "split_missingness",
                "split": split,
                "ltv_band": "ALL",
                "rows": len(group),
                "missing_ltv_rows": int(group["ltv_proxy"].isna().sum()),
                "missing_ltv_rate": group["ltv_proxy"].isna().mean(),
                "mean_ltv": group["ltv_proxy"].mean(),
                "median_ltv": group["ltv_proxy"].median(),
                "actual_cure_lgd": group["realized_lgd_model_target"].mean(),
                "predicted_cure_lgd": group["predicted_cure_lgd"].mean(),
            }
        )
        for band, band_group in group.groupby("ltv_band", observed=True):
            rows.append(
                {
                    "section": "ltv_band",
                    "split": split,
                    "ltv_band": band,
                    "rows": len(band_group),
                    "missing_ltv_rows": int(band_group["ltv_proxy"].isna().sum()),
                    "missing_ltv_rate": band_group["ltv_proxy"].isna().mean(),
                    "mean_ltv": band_group["ltv_proxy"].mean(),
                    "median_ltv": band_group["ltv_proxy"].median(),
                    "actual_cure_lgd": band_group["realized_lgd_model_target"].mean(),
                    "predicted_cure_lgd": band_group["predicted_cure_lgd"].mean(),
                }
            )
    relation = cure[["ltv_proxy", "realized_lgd_model_target"]].dropna()
    rows.append(
        {
            "section": "relationship",
            "split": "ALL",
            "ltv_band": "correlation",
            "rows": len(relation),
            "missing_ltv_rows": int(cure["ltv_proxy"].isna().sum()),
            "missing_ltv_rate": cure["ltv_proxy"].isna().mean(),
            "mean_ltv": relation["ltv_proxy"].mean(),
            "median_ltv": relation["ltv_proxy"].median(),
            "actual_cure_lgd": relation["realized_lgd_model_target"].mean(),
            "predicted_cure_lgd": relation["ltv_proxy"].corr(
                relation["realized_lgd_model_target"]
            ),
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
