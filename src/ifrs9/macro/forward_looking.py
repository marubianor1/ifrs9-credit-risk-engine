"""Forward-looking IFRS 9 PD scenarios using Vasicek and macro satellite models."""

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

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.macro.config import ForwardLookingConfig, load_forward_looking_config
from ifrs9.macro.data import load_macro_dataset


@dataclass(frozen=True)
class ForwardLookingRunResult:
    """Result of a forward-looking PD scenario run."""

    run_id: str
    parent_pd_run: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


@dataclass(frozen=True)
class ScenarioResult:
    """Result of a scenario simulation."""

    macro_paths: pd.DataFrame
    scenario_pd_curves: pd.DataFrame
    weighted_pd_curves: pd.DataFrame


def vasicek_pit_pd(
    ttc_pd: float | np.ndarray,
    systematic_factor: float | np.ndarray,
    rho: float,
) -> np.ndarray:
    """Calculate PIT PD from TTC PD, systematic factor, and asset correlation."""
    _validate_rho(rho)
    clipped = np.clip(ttc_pd, 1e-8, 1 - 1e-8)
    numerator = norm.ppf(clipped) - np.sqrt(rho) * systematic_factor
    denominator = np.sqrt(1 - rho)
    return norm.cdf(numerator / denominator)


def infer_systematic_factor(
    observed_default_rate: float | np.ndarray,
    ttc_pd: float | np.ndarray,
    rho: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Infer historical systematic factor from observed default experience."""
    _validate_rho(rho)
    observed = np.clip(observed_default_rate, epsilon, 1 - epsilon)
    ttc = np.clip(ttc_pd, epsilon, 1 - epsilon)
    return (norm.ppf(ttc) - np.sqrt(1 - rho) * norm.ppf(observed)) / np.sqrt(rho)


def calibrate_rho(
    observed_default_rate: np.ndarray,
    ttc_pd: float,
    config: ForwardLookingConfig,
) -> tuple[float, pd.DataFrame]:
    """Estimate rho by matching inferred factor volatility to one."""
    if config.vasicek.rho_override is not None:
        rho = config.vasicek.rho_override
        _validate_rho(rho)
        return rho, pd.DataFrame(
            [{"rho": rho, "factor_std": np.nan, "objective": 0.0, "selected": True}]
        )
    rows = []
    for rho in np.linspace(
        config.vasicek.rho_min,
        config.vasicek.rho_max,
        config.vasicek.rho_grid_size,
    ):
        factor = infer_systematic_factor(
            observed_default_rate,
            ttc_pd,
            float(rho),
            config.vasicek.clipping_epsilon,
        )
        factor_std = float(np.nanstd(factor, ddof=1))
        rows.append(
            {
                "rho": float(rho),
                "factor_std": factor_std,
                "objective": abs(factor_std - 1.0),
            }
        )
    grid = pd.DataFrame(rows)
    selected_index = grid["objective"].idxmin()
    grid["selected"] = False
    grid.loc[selected_index, "selected"] = True
    return float(grid.loc[selected_index, "rho"]), grid


def simulate_macro_scenario(
    config: ForwardLookingConfig,
    shocks: dict[str, dict[str, float]],
    *,
    repo_root: Path | None = None,
    parent_pd_run: str | None = None,
) -> ScenarioResult:
    """Simulate configured macro shocks without refitting historical models."""
    root = find_repo_root(repo_root)
    parent = parent_pd_run or config.parent_pd_run
    pd_data = _load_parent_pd_outputs(root, parent)
    macro = load_macro_dataset(root, config)
    historical = _historical_factor_table(pd_data.predictions, pd_data.portfolio_ttc, config)
    rho, _ = calibrate_rho(
        historical["observed_default_rate"].to_numpy(),
        pd_data.portfolio_ttc,
        config,
    )
    factor, satellite, _ = _fit_macro_models(historical, macro, config)
    scenario_config = config.model_copy(
        deep=True,
        update={"scenarios": config.scenarios.model_copy(update={"shocks": shocks})},
    )
    scenario_paths = _scenario_macro_paths(macro, scenario_config)
    factor_paths = _predict_scenario_factors(scenario_paths, factor, satellite, config)
    scenario_pd = _scenario_rating_pd_curves(pd_data.rating_ttc, factor_paths, rho)
    weighted = _weighted_pd_curves(scenario_pd)
    return ScenarioResult(factor_paths, scenario_pd, weighted)


def run_forward_looking(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    pd_run_id: str | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> ForwardLookingRunResult:
    """Run the forward-looking PD scenario framework."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_forward_looking_config(root, config_path)
    parent = pd_run_id or config.parent_pd_run
    run_id = run_id or f"forward_looking_{parent}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.run.artifact_root / run_id
    model_dir = root / config.run.model_root / run_id
    if not force and (artifact_dir.exists() or model_dir.exists()):
        msg = f"Forward-looking run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    pd_data = _load_parent_pd_outputs(root, parent)
    macro = load_macro_dataset(root, config)
    historical = _historical_factor_table(pd_data.predictions, pd_data.portfolio_ttc, config)
    rho, rho_grid = calibrate_rho(
        historical["observed_default_rate"].to_numpy(),
        pd_data.portfolio_ttc,
        config,
    )
    historical["systematic_factor"] = infer_systematic_factor(
        historical["observed_default_rate"],
        historical["ttc_pd"],
        rho,
        config.vasicek.clipping_epsilon,
    )
    factor_model, satellite_model, diagnostics = _fit_macro_models(historical, macro, config)
    backtesting = _historical_backtesting(
        historical,
        macro,
        pd_data.portfolio_ttc,
        rho,
        factor_model,
        satellite_model,
        config,
    )
    scenario_paths = _scenario_macro_paths(macro, config)
    factor_paths = _predict_scenario_factors(scenario_paths, factor_model, satellite_model, config)
    scenario_pd = _scenario_rating_pd_curves(pd_data.rating_ttc, factor_paths, rho)
    weighted_pd = _weighted_pd_curves(scenario_pd)
    scenario_lifetime = _scenario_lifetime_curves(
        pd_data.lifetime,
        scenario_pd,
        config,
    )
    weighted_lifetime = _weighted_lifetime_curves(scenario_lifetime)

    macro.to_csv(artifact_dir / "macro_quarterly.csv", index=False)
    historical.to_csv(artifact_dir / "historical_systematic_factor.csv", index=False)
    rho_grid.to_csv(artifact_dir / "rho_calibration_grid.csv", index=False)
    diagnostics.coefficients.to_csv(artifact_dir / "macro_model_coefficients.csv", index=False)
    diagnostics.residuals.to_csv(artifact_dir / "macro_model_residuals.csv", index=False)
    pd.DataFrame([diagnostics.metrics]).to_csv(
        artifact_dir / "macro_model_diagnostics.csv",
        index=False,
    )
    scenario_paths.to_csv(artifact_dir / "scenario_macro_paths.csv", index=False)
    factor_paths.to_csv(artifact_dir / "scenario_systematic_factors.csv", index=False)
    scenario_pd.to_csv(artifact_dir / "scenario_pd_curves.csv", index=False)
    weighted_pd.to_csv(artifact_dir / "weighted_pd_curves.csv", index=False)
    scenario_lifetime.to_csv(artifact_dir / "scenario_lifetime_curves.csv", index=False)
    weighted_lifetime.to_csv(artifact_dir / "weighted_lifetime_curves.csv", index=False)
    backtesting.quarterly.to_csv(artifact_dir / "historical_backtesting.csv", index=False)
    pd.DataFrame([backtesting.metrics]).to_csv(
        artifact_dir / "historical_backtesting_metrics.csv",
        index=False,
    )
    backtesting.stress.to_csv(artifact_dir / "stress_2019_2021_backtesting.csv", index=False)
    _write_json(
        artifact_dir / "scenario_weights.json",
        {"weights": config.scenarios.weights, "sum": sum(config.scenarios.weights.values())},
    )
    with (model_dir / "forward_looking_models.pkl").open("wb") as stream:
        pickle.dump(
            {
                "rho": rho,
                "factor_model": factor_model,
                "satellite_model": satellite_model,
                "feature_columns": config.macro.feature_columns,
            },
            stream,
        )

    result = ForwardLookingRunResult(
        run_id=run_id,
        parent_pd_run=parent,
        artifact_path=str(artifact_dir),
        model_path=str(model_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "parent_pd_run": parent,
        "config": config.model_dump(),
        "rho": rho,
        "macro_data_version": _macro_data_version(config),
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


@dataclass(frozen=True)
class _PDOutputs:
    predictions: pd.DataFrame
    portfolio_ttc: float
    rating_ttc: pd.DataFrame
    lifetime: pd.DataFrame


@dataclass(frozen=True)
class _ModelDiagnostics:
    coefficients: pd.DataFrame
    residuals: pd.DataFrame
    metrics: dict[str, float]


@dataclass(frozen=True)
class _BacktestingOutputs:
    quarterly: pd.DataFrame
    metrics: dict[str, float]
    stress: pd.DataFrame


def _load_parent_pd_outputs(repo_root: Path, pd_run_id: str) -> _PDOutputs:
    base = repo_root / "artifacts" / "pd" / pd_run_id
    if not base.exists():
        msg = f"Parent PD run not found: {base}"
        raise FileNotFoundError(msg)
    predictions = pd.read_parquet(base / "pd_predictions.parquet")
    predictions["as_of_date"] = pd.to_datetime(predictions["as_of_date"])
    portfolio_ttc = float(pd.read_csv(base / "portfolio_ttc.csv")["portfolio_ttc_pd"].iloc[0])
    rating_ttc = pd.read_csv(base / "rating_ttc.csv")
    lifetime = pd.read_csv(base / "lifetime_curves.csv")
    return _PDOutputs(predictions, portfolio_ttc, rating_ttc, lifetime)


def _historical_factor_table(
    predictions: pd.DataFrame,
    ttc_pd: float,
    config: ForwardLookingConfig,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = frame["as_of_date"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    grouped = (
        frame.groupby("date", observed=True)
        .agg(observed_default_rate=("target", "mean"), observations=("target", "size"))
        .reset_index()
    )
    grouped["ttc_pd"] = ttc_pd
    mask = (
        (grouped["date"] >= pd.Timestamp(config.macro.start_date))
        & (grouped["date"] <= pd.Timestamp(config.backtesting.stress_end))
    )
    return grouped.loc[mask].reset_index(drop=True)


def _fit_macro_models(
    historical: pd.DataFrame,
    macro: pd.DataFrame,
    config: ForwardLookingConfig,
) -> tuple[LinearRegression, Ridge, _ModelDiagnostics]:
    joined = _joined_model_frame(historical, macro, config)
    features = config.macro.feature_columns
    factor_features = _economic_strength_features(joined[features])
    factor_model = LinearRegression(positive=True)
    factor_model.fit(factor_features, joined["systematic_factor"])
    satellite_model = Ridge(alpha=1 / config.satellite.regularization_strength)
    satellite_model.fit(
        joined[features],
        logit(np.clip(joined["observed_default_rate"], 1e-6, 1 - 1e-6)),
    )
    factor_prediction = factor_model.predict(factor_features)
    satellite_prediction = expit(satellite_model.predict(joined[features]))
    residuals = joined[["date", "observed_default_rate", "systematic_factor"]].copy()
    residuals["vasicek_factor_prediction"] = factor_prediction
    residuals["factor_residual"] = residuals["systematic_factor"] - factor_prediction
    residuals["satellite_pd"] = satellite_prediction
    residuals["satellite_residual"] = residuals["observed_default_rate"] - satellite_prediction
    coefficients = _coefficient_table(factor_model, satellite_model, joined[features], features)
    metrics = _macro_diagnostics(
        joined["systematic_factor"].to_numpy(),
        factor_prediction,
        joined["observed_default_rate"].to_numpy(),
        satellite_prediction,
        residuals["factor_residual"].to_numpy(),
        joined[features],
    )
    return factor_model, satellite_model, _ModelDiagnostics(coefficients, residuals, metrics)


def _joined_model_frame(
    historical: pd.DataFrame,
    macro: pd.DataFrame,
    config: ForwardLookingConfig,
) -> pd.DataFrame:
    if "systematic_factor" not in historical.columns:
        historical = historical.copy()
        rho, _ = calibrate_rho(
            historical["observed_default_rate"].to_numpy(),
            float(historical["ttc_pd"].iloc[0]),
            config,
        )
        historical["systematic_factor"] = infer_systematic_factor(
            historical["observed_default_rate"],
            historical["ttc_pd"],
            rho,
            config.vasicek.clipping_epsilon,
        )
    joined = historical.merge(macro, on="date", how="inner")
    return joined.dropna(subset=config.macro.feature_columns + ["systematic_factor"])


def _coefficient_table(
    factor_model: LinearRegression,
    satellite_model: Ridge,
    features: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = [
        {
            "model": "vasicek_factor",
            "feature": "INTERCEPT",
            "coefficient": float(factor_model.intercept_),
            "vif": np.nan,
        },
        {
            "model": "macro_satellite",
            "feature": "INTERCEPT",
            "coefficient": float(satellite_model.intercept_),
            "vif": np.nan,
        },
    ]
    vifs = _variance_inflation_factors(features)
    for feature, coefficient in zip(feature_names, factor_model.coef_, strict=False):
        rows.append(
            {
                "model": "vasicek_factor",
                "feature": feature,
                "coefficient": float(coefficient * _economic_strength_sign(feature)),
                "vif": vifs.get(feature, np.nan),
            }
        )
    for feature, coefficient in zip(feature_names, satellite_model.coef_, strict=False):
        rows.append(
            {
                "model": "macro_satellite",
                "feature": feature,
                "coefficient": float(coefficient),
                "vif": vifs.get(feature, np.nan),
            }
        )
    return pd.DataFrame(rows)


def _economic_strength_sign(feature: str) -> int:
    if feature in {
        "unemployment_rate",
        "unemployment_rate_change",
        "mortgage_rate",
        "mortgage_rate_change",
    }:
        return -1
    return 1


def _variance_inflation_factors(features: pd.DataFrame) -> dict[str, float]:
    output = {}
    for column in features.columns:
        others = features.drop(columns=[column])
        if others.empty:
            output[column] = 1.0
            continue
        model = LinearRegression().fit(others, features[column])
        r2 = model.score(others, features[column])
        output[column] = float(1 / max(1 - r2, 1e-6))
    return output


def _macro_diagnostics(
    factor_actual: np.ndarray,
    factor_prediction: np.ndarray,
    observed_pd: np.ndarray,
    satellite_prediction: np.ndarray,
    factor_residuals: np.ndarray,
    features: pd.DataFrame,
) -> dict[str, float]:
    residual_series = pd.Series(factor_residuals)
    return {
        "factor_rmse": float(mean_squared_error(factor_actual, factor_prediction) ** 0.5),
        "factor_mae": float(mean_absolute_error(factor_actual, factor_prediction)),
        "satellite_rmse": float(mean_squared_error(observed_pd, satellite_prediction) ** 0.5),
        "satellite_mae": float(mean_absolute_error(observed_pd, satellite_prediction)),
        "satellite_brier": float(np.mean((observed_pd - satellite_prediction) ** 2)),
        "factor_residual_autocorrelation_lag1": float(residual_series.autocorr(lag=1)),
        "max_vif": float(max(_variance_inflation_factors(features).values())),
    }


def _historical_backtesting(
    historical: pd.DataFrame,
    macro: pd.DataFrame,
    ttc_pd: float,
    rho: float,
    factor_model: LinearRegression,
    satellite_model: Ridge,
    config: ForwardLookingConfig,
) -> _BacktestingOutputs:
    joined = _joined_model_frame(historical, macro, config)
    features = joined[config.macro.feature_columns]
    joined["vasicek_systematic_factor"] = factor_model.predict(
        _economic_strength_features(features)
    )
    joined["vasicek_pit_pd"] = vasicek_pit_pd(ttc_pd, joined["vasicek_systematic_factor"], rho)
    joined["satellite_pit_pd"] = expit(satellite_model.predict(features))
    joined["vasicek_residual"] = joined["observed_default_rate"] - joined["vasicek_pit_pd"]
    joined["satellite_residual"] = joined["observed_default_rate"] - joined["satellite_pit_pd"]
    observed = joined["observed_default_rate"]
    vasicek_pd = joined["vasicek_pit_pd"]
    satellite_pd = joined["satellite_pit_pd"]
    metrics = {
        "vasicek_rmse": float(mean_squared_error(observed, vasicek_pd) ** 0.5),
        "vasicek_mae": float(mean_absolute_error(observed, vasicek_pd)),
        "vasicek_brier": float(np.mean((observed - vasicek_pd) ** 2)),
        "satellite_rmse": float(mean_squared_error(observed, satellite_pd) ** 0.5),
        "satellite_mae": float(mean_absolute_error(observed, satellite_pd)),
        "satellite_brier": float(np.mean((observed - satellite_pd) ** 2)),
    }
    stress_mask = (
        (joined["date"] >= pd.Timestamp(config.backtesting.stress_start))
        & (joined["date"] <= pd.Timestamp(config.backtesting.stress_end))
    )
    return _BacktestingOutputs(joined, metrics, joined.loc[stress_mask].copy())


def _scenario_macro_paths(macro: pd.DataFrame, config: ForwardLookingConfig) -> pd.DataFrame:
    latest = macro.sort_values("date").dropna(subset=config.macro.feature_columns).iloc[-1]
    dates = pd.date_range(
        pd.Timestamp(config.scenarios.start_date),
        periods=config.scenarios.horizon_quarters,
        freq="QE",
    )
    rows = []
    for scenario, shock in config.scenarios.shocks.items():
        for step, date in enumerate(dates, start=1):
            fraction = step / config.scenarios.horizon_quarters
            row: dict[str, Any] = {
                "scenario": scenario,
                "date": date,
                "scenario_weight": config.scenarios.weights[scenario],
            }
            for column in config.macro.feature_columns:
                base_column = _base_feature(column)
                base_value = float(latest[column])
                if column.endswith("_change"):
                    row[column] = (
                        float(shock.get(base_column, 0.0)) / config.scenarios.horizon_quarters
                    )
                else:
                    row[column] = base_value + float(shock.get(column, 0.0)) * fraction
            rows.append(row)
    return pd.DataFrame(rows)


def _base_feature(column: str) -> str:
    if column.endswith("_change"):
        return column.removesuffix("_change")
    return column


def _economic_strength_features(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    weaker_when_higher = {
        "unemployment_rate",
        "unemployment_rate_change",
        "mortgage_rate",
        "mortgage_rate_change",
    }
    for column in weaker_when_higher.intersection(output.columns):
        output[column] = -output[column]
    return output


def _predict_scenario_factors(
    scenario_paths: pd.DataFrame,
    factor_model: LinearRegression,
    satellite_model: Ridge,
    config: ForwardLookingConfig,
) -> pd.DataFrame:
    frame = scenario_paths.copy()
    features = frame[config.macro.feature_columns]
    frame["systematic_factor"] = factor_model.predict(_economic_strength_features(features))
    frame["satellite_portfolio_pd"] = expit(satellite_model.predict(features))
    return frame


def _scenario_rating_pd_curves(
    rating_ttc: pd.DataFrame,
    scenario_factors: pd.DataFrame,
    rho: float,
) -> pd.DataFrame:
    rows = []
    for factor_row in scenario_factors.itertuples(index=False):
        for rating_row in rating_ttc.itertuples(index=False):
            pit_pd = float(
                vasicek_pit_pd(
                    rating_row.rating_ttc_pd,
                    factor_row.systematic_factor,
                    rho,
                )
            )
            rows.append(
                {
                    "scenario": factor_row.scenario,
                    "date": factor_row.date,
                    "rating": rating_row.rating,
                    "ttc_pd": rating_row.rating_ttc_pd,
                    "systematic_factor": factor_row.systematic_factor,
                    "pit_pd_12m": pit_pd,
                    "scenario_weight": factor_row.scenario_weight,
                    "weighted_pd_12m": pit_pd * factor_row.scenario_weight,
                }
            )
    return pd.DataFrame(rows)


def _weighted_pd_curves(scenario_pd: pd.DataFrame) -> pd.DataFrame:
    return (
        scenario_pd.groupby(["date", "rating"], observed=True)
        .agg(
            ttc_pd=("ttc_pd", "first"),
            weighted_pd_12m=("weighted_pd_12m", "sum"),
            scenario_weight_sum=("scenario_weight", "sum"),
        )
        .reset_index()
    )


def _scenario_lifetime_curves(
    base_lifetime: pd.DataFrame,
    scenario_pd: pd.DataFrame,
    config: ForwardLookingConfig,
) -> pd.DataFrame:
    curves = base_lifetime[base_lifetime["month"] <= config.lifetime.max_horizon_months].copy()
    rows = []
    for pd_row in scenario_pd.itertuples(index=False):
        rating_curve = curves[curves["rating"] == pd_row.rating].copy()
        factor = _hazard_scale_factor(
            rating_curve.loc[rating_curve["month"] <= 12, "conditional_pd"].to_list(),
            pd_row.pit_pd_12m,
        )
        survival = 1.0
        cumulative = 0.0
        for curve_row in rating_curve.itertuples(index=False):
            conditional = min(1.0, float(curve_row.conditional_pd) * factor)
            marginal = survival * conditional
            cumulative = min(1.0, cumulative + marginal)
            survival = max(0.0, survival - marginal)
            rows.append(
                {
                    "scenario": pd_row.scenario,
                    "date": pd_row.date,
                    "rating": pd_row.rating,
                    "month": int(curve_row.month),
                    "conditional_pd": conditional,
                    "hazard": conditional,
                    "marginal_pd": marginal,
                    "cumulative_pd": cumulative,
                    "survival_probability": survival,
                    "scenario_weight": pd_row.scenario_weight,
                    "weighted_marginal_pd": marginal * pd_row.scenario_weight,
                    "weighted_cumulative_pd": cumulative * pd_row.scenario_weight,
                }
            )
    return pd.DataFrame(rows)


def _weighted_lifetime_curves(scenario_lifetime: pd.DataFrame) -> pd.DataFrame:
    weighted = (
        scenario_lifetime.groupby(["date", "rating", "month"], observed=True)
        .agg(
            marginal_pd=("weighted_marginal_pd", "sum"),
            cumulative_pd=("weighted_cumulative_pd", "sum"),
            scenario_weight_sum=("scenario_weight", "sum"),
        )
        .reset_index()
        .sort_values(["date", "rating", "month"])
    )
    weighted["survival_probability"] = 1 - weighted["cumulative_pd"]
    weighted["conditional_pd"] = weighted.groupby(["date", "rating"], observed=True).apply(
        _conditional_from_weighted,
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True)
    weighted["hazard"] = weighted["conditional_pd"]
    return weighted


def _conditional_from_weighted(group: pd.DataFrame) -> pd.Series:
    previous_survival = 1.0
    values = []
    for row in group.itertuples(index=False):
        values.append(row.marginal_pd / previous_survival if previous_survival > 0 else 0.0)
        previous_survival = max(0.0, 1 - row.cumulative_pd)
    return pd.Series(values, index=group.index)


def _hazard_scale_factor(hazards: list[float], target_cumulative: float) -> float:
    target = min(max(target_cumulative, 0.0), 1.0)
    if not hazards or target == 0:
        return 0.0
    low = 0.0
    high = 1.0
    while _scaled_cumulative(hazards, high) < target and high < 1_000:
        high *= 2
    for _ in range(60):
        midpoint = (low + high) / 2
        if _scaled_cumulative(hazards, midpoint) < target:
            low = midpoint
        else:
            high = midpoint
    return high


def _scaled_cumulative(hazards: list[float], factor: float) -> float:
    survival = 1.0
    cumulative = 0.0
    for hazard in hazards:
        conditional = min(1.0, hazard * factor)
        marginal = survival * conditional
        cumulative = min(1.0, cumulative + marginal)
        survival = max(0.0, survival - marginal)
    return cumulative


def _validate_rho(rho: float) -> None:
    if not 0 < rho < 1:
        msg = f"rho must be between 0 and 1, got {rho}"
        raise ValueError(msg)


def _macro_data_version(config: ForwardLookingConfig) -> dict[str, str]:
    return {name: series.fred_id for name, series in config.macro.series.items()}


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


def load_forward_looking_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted forward-looking run manifest."""
    config = load_forward_looking_config(repo_root)
    return json.loads((repo_root / config.run.artifact_root / run_id / "run.json").read_text())
