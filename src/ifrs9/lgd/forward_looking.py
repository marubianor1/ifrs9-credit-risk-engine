"""Forward-looking LGD macro overlay framework."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ifrs9.ingestion.schema_registry import find_repo_root


class LGDForwardOutputConfig(BaseModel):
    """Artifact path configuration."""

    artifact_root: str
    model_root: str


class LGDForwardSensitivityConfig(BaseModel):
    """Deterministic macro sensitivity shocks."""

    hpi_shocks: list[float]
    unemployment_shocks: list[float]


class LGDForwardLookingConfig(BaseModel):
    """Configurable forward-looking LGD overlay settings."""

    version: str
    method: Literal["macro_overlay"]
    parent_lgd_run: str
    scenario_run: str
    calibration_split: Literal["VALIDATION"]
    overlay_floor: float = Field(ge=0)
    overlay_cap: float | None = Field(default=None, gt=0)
    scenario_horizon_quarters: int = Field(ge=1)
    macro_features: list[str]
    stress_weights: dict[str, float]
    sensitivity: LGDForwardSensitivityConfig
    output: LGDForwardOutputConfig

    @model_validator(mode="after")
    def validate_overlay_bounds(self) -> LGDForwardLookingConfig:
        """Validate overlay clipping bounds and stress feature coverage."""
        if self.overlay_cap is not None and self.overlay_floor >= self.overlay_cap:
            msg = "overlay_floor must be below overlay_cap"
            raise ValueError(msg)
        missing = set(self.macro_features).difference(self.stress_weights)
        if missing:
            msg = f"Missing stress weights for macro features: {sorted(missing)}"
            raise ValueError(msg)
        return self


class LGDForwardLookingConfigFile(BaseModel):
    """Top-level YAML shape."""

    lgd_forward_looking: LGDForwardLookingConfig


@dataclass(frozen=True)
class LGDForwardLookingResult:
    """Result of a forward-looking LGD run."""

    run_id: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


@dataclass(frozen=True)
class LGDScenarioResult:
    """Result of a deterministic LGD scenario simulation."""

    scenario_lgd: pd.DataFrame
    weighted_lgd: pd.DataFrame
    sensitivity: pd.DataFrame


def load_lgd_forward_config(
    repo_root: Path,
    path: Path | None = None,
) -> LGDForwardLookingConfig:
    """Load forward-looking LGD configuration."""
    config_path = path or repo_root / "config" / "lgd_forward_looking.yaml"
    with config_path.open() as stream:
        parsed = LGDForwardLookingConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.lgd_forward_looking


def run_lgd_forward_looking(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> LGDForwardLookingResult:
    """Run the forward-looking LGD scenario overlay."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_lgd_forward_config(root, config_path)
    run_id = run_id or f"lgd_fl_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.output.artifact_root / run_id
    model_dir = root / config.output.model_root / run_id
    if not force and (artifact_dir.exists() or model_dir.exists()):
        msg = f"LGD forward-looking run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_lgd_episodes(root, config.parent_lgd_run)
    macro = _load_historical_macro(root, config.scenario_run)
    historical = _historical_lgd_regime_dataset(episodes, macro, config)
    overlay = _fit_overlay(historical, config)
    scored = _apply_historical_overlay(episodes, macro, overlay, config)
    scenario_result = simulate_lgd_scenario(
        config,
        repo_root=root,
        overlay=overlay,
        episodes=episodes,
    )

    historical.to_csv(artifact_dir / "historical_lgd_regime_dataset.csv", index=False)
    _relationship_diagnostics(historical, config).to_csv(
        artifact_dir / "macro_lgd_relationships.csv",
        index=False,
    )
    _coefficient_table(overlay).to_csv(artifact_dir / "overlay_coefficients.csv", index=False)
    _historical_backtest(scored).to_csv(artifact_dir / "historical_backtest.csv", index=False)
    _historical_backtest(scored, ["split", "rating_at_default"]).to_csv(
        artifact_dir / "historical_backtest_by_rating.csv",
        index=False,
    )
    _historical_backtest(scored, ["split", "default_year"]).to_csv(
        artifact_dir / "historical_backtest_by_default_year.csv",
        index=False,
    )
    _oot_cure_lgd_diagnostics(scored).to_csv(
        artifact_dir / "oot_cure_lgd_diagnostics.csv",
        index=False,
    )
    _scenario_ordering_diagnostics(scenario_result.scenario_lgd).to_csv(
        artifact_dir / "scenario_ordering_diagnostics.csv",
        index=False,
    )
    scenario_result.scenario_lgd.to_csv(artifact_dir / "scenario_lgd_by_rating.csv", index=False)
    scenario_result.weighted_lgd.to_csv(artifact_dir / "probability_weighted_lgd.csv", index=False)
    scenario_result.sensitivity.to_csv(artifact_dir / "scenario_sensitivity.csv", index=False)
    _write_json(artifact_dir / "config_snapshot.json", config.model_dump())

    with (model_dir / "lgd_forward_overlay.pkl").open("wb") as stream:
        pickle.dump({"overlay": overlay, "config": config.model_dump()}, stream)

    result = LGDForwardLookingResult(
        run_id=run_id,
        artifact_path=str(artifact_dir),
        model_path=str(model_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "config": config.model_dump(),
        "parent_lgd_run": config.parent_lgd_run,
        "scenario_run": config.scenario_run,
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
        "rows": len(scored),
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


def simulate_lgd_scenario(
    config: LGDForwardLookingConfig,
    overrides: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
    overlay: dict[str, Any] | None = None,
    episodes: pd.DataFrame | None = None,
) -> LGDScenarioResult:
    """Apply a fitted LGD macro overlay to configured scenario paths."""
    root = find_repo_root(repo_root)
    base_episodes = (
        episodes if episodes is not None else _load_lgd_episodes(root, config.parent_lgd_run)
    )
    fitted = overlay or _load_overlay_model(root, config)
    scenario_macro = _load_scenario_macro(root, config.scenario_run, config).copy()
    if overrides:
        for column, shock in overrides.items():
            if column in scenario_macro.columns:
                scenario_macro[column] = scenario_macro[column] + float(shock)
    cure_model = _load_cure_lgd_model(root, config.parent_lgd_run)
    if cure_model["model"] is not None:
        scenario_lgd = _scenario_lgd_from_stressed_ltv(
            base_episodes,
            scenario_macro,
            cure_model,
        )
    else:
        rating_base = _rating_structural_base(base_episodes)
        scenario_lgd = _scenario_lgd_by_rating(rating_base, scenario_macro, fitted, config)
    weighted = _weighted_scenario_lgd(scenario_lgd)
    sensitivity = _scenario_sensitivity(
        base_episodes,
        scenario_macro,
        fitted,
        config,
        cure_model,
    )
    return LGDScenarioResult(scenario_lgd, weighted, sensitivity)


def load_lgd_forward_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted forward-looking LGD run manifest."""
    config = load_lgd_forward_config(repo_root)
    return json.loads((repo_root / config.output.artifact_root / run_id / "run.json").read_text())


def _load_lgd_episodes(repo_root: Path, run_id: str) -> pd.DataFrame:
    path = repo_root / "artifacts" / "lgd" / run_id / "lgd_episodes.parquet"
    if not path.exists():
        msg = f"Parent LGD episodes not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(path)
    frame["default_date"] = pd.to_datetime(frame["default_date"])
    return frame[frame["resolved_flag"] & frame["ead_at_default"].gt(0)].copy()


def _load_historical_macro(repo_root: Path, scenario_run: str) -> pd.DataFrame:
    base = repo_root / "artifacts" / "forward_looking" / scenario_run
    macro = pd.read_csv(base / "macro_quarterly.csv")
    factor = pd.read_csv(base / "historical_systematic_factor.csv")
    macro["date"] = pd.to_datetime(macro["date"])
    factor["date"] = pd.to_datetime(factor["date"])
    return macro.merge(
        factor[["date", "systematic_factor"]],
        on="date",
        how="left",
    )


def _load_scenario_macro(
    repo_root: Path,
    scenario_run: str,
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    path = (
        repo_root
        / "artifacts"
        / "forward_looking"
        / scenario_run
        / "scenario_systematic_factors.csv"
    )
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["scenario", "date"]).groupby("scenario", observed=True).head(
        config.scenario_horizon_quarters
    )


def _historical_lgd_regime_dataset(
    episodes: pd.DataFrame,
    macro: pd.DataFrame,
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    frame = _episodes_with_macro(episodes, macro, config)
    frame["default_quarter"] = frame["default_date"].dt.to_period("Q").dt.to_timestamp(
        how="end"
    ).dt.normalize()
    frame["cure_lgd_observed"] = np.where(
        frame["cured_flag"],
        frame["realized_lgd_model_target"],
        np.nan,
    )
    frame["non_cure_lgd_observed"] = np.where(
        ~frame["cured_flag"],
        frame["realized_lgd_model_target"],
        np.nan,
    )
    grouped = (
        frame.groupby(["default_quarter", "split"], observed=True)
        .agg(
            rows=("loan_id", "size"),
            realized_lgd=("realized_lgd_model_target", "mean"),
            predicted_lgd=("predicted_lgd", "mean"),
            cure_rate=("cured_flag", "mean"),
            cure_lgd=("cure_lgd_observed", "mean"),
            non_cure_lgd=("non_cure_lgd_observed", "mean"),
            average_ltv=("estimated_loan_to_value_at_default", "mean"),
            average_current_upb_ratio=("current_upb_to_original_upb_at_default", "mean"),
            delinquency_default_share=("default_due_to_delinquency", "mean"),
            credit_event_default_share=("default_due_to_credit_event", "mean"),
            r5_share=("rating_at_default", lambda x: float((x == "R5").mean())),
        )
        .reset_index()
        .rename(columns={"default_quarter": "date"})
    )
    macro_cols = ["date", *config.macro_features]
    grouped = grouped.merge(macro[macro_cols], on="date", how="left")
    grouped["structural_oe_ratio"] = grouped["realized_lgd"] / grouped["predicted_lgd"]
    grouped["stress_index"] = _stress_index(grouped, config, fitted=None)
    return grouped


def _episodes_with_macro(
    episodes: pd.DataFrame,
    macro: pd.DataFrame,
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    frame = episodes.copy()
    frame["date"] = (
        frame["default_date"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    )
    return frame.merge(macro[["date", *config.macro_features]], on="date", how="left")


def _fit_overlay(historical: pd.DataFrame, config: LGDForwardLookingConfig) -> dict[str, Any]:
    train = historical[historical["split"] == "TRAIN"].dropna(
        subset=["structural_oe_ratio", "stress_index"]
    )
    validation = historical[historical["split"] == config.calibration_split].dropna(
        subset=["structural_oe_ratio", "stress_index"]
    )
    if len(train) < 2:
        slope = 0.0
        train_intercept = 0.0
    else:
        x = train["stress_index"].to_numpy(dtype=float)
        y = np.log(np.clip(train["structural_oe_ratio"].to_numpy(dtype=float), 1e-6, None))
        slope = max(float(np.cov(x, y, ddof=0)[0, 1] / max(np.var(x), 1e-12)), 0.0)
        train_intercept = float(y.mean() - slope * x.mean())
    if validation.empty:
        intercept = train_intercept
    else:
        predicted_no_intercept = np.exp(slope * validation["stress_index"])
        numerator = (validation["realized_lgd"] * validation["rows"]).sum()
        denominator = (
            validation["predicted_lgd"] * predicted_no_intercept * validation["rows"]
        ).sum()
        intercept = float(np.log(max(numerator, 1e-6) / max(denominator, 1e-6)))
    return {
        "method": "multiplicative_macro_stress_overlay",
        "slope": slope,
        "train_intercept": train_intercept,
        "validation_intercept": intercept,
        "stress_mean": historical.loc[historical["split"] == "TRAIN", config.macro_features]
        .mean(numeric_only=True)
        .to_dict(),
        "stress_std": historical.loc[historical["split"] == "TRAIN", config.macro_features]
        .std(numeric_only=True)
        .replace(0, 1)
        .fillna(1)
        .to_dict(),
        "stress_weights": config.stress_weights,
        "selection_basis": "TRAIN slope, VALIDATION intercept, OOT held out",
    }


def _apply_historical_overlay(
    episodes: pd.DataFrame,
    macro: pd.DataFrame,
    overlay: dict[str, Any],
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    scored = _episodes_with_macro(episodes, macro, config)
    scored["macro_overlay"] = _macro_multiplier(scored, overlay, config)
    calibration = scored.get(
        "expected_lgd_calibration_factor",
        pd.Series(1.0, index=scored.index),
    )
    scored["cure_lgd_forward_adjusted"] = _clip_lgd(
        scored["predicted_cure_lgd"] * calibration * scored["macro_overlay"],
    )
    scored["non_cure_lgd_forward_adjusted"] = _clip_lgd(
        scored["predicted_non_cure_lgd"] * calibration * scored["macro_overlay"],
    )
    scored["lgd_forward_adjusted"] = (
        scored["predicted_cure_probability"] * scored["cure_lgd_forward_adjusted"]
        + (1 - scored["predicted_cure_probability"]) * scored["non_cure_lgd_forward_adjusted"]
    )
    scored["branch_reconciled_lgd"] = scored["lgd_forward_adjusted"]
    return scored


def _stress_index(
    frame: pd.DataFrame,
    config: LGDForwardLookingConfig,
    fitted: dict[str, Any] | None,
) -> pd.Series:
    if fitted is None:
        mean = frame[config.macro_features].mean(numeric_only=True).to_dict()
        std = frame[config.macro_features].std(numeric_only=True).replace(0, 1).fillna(1).to_dict()
    else:
        mean = fitted["stress_mean"]
        std = fitted["stress_std"]
    output = pd.Series(0.0, index=frame.index)
    for feature in config.macro_features:
        z = (frame[feature] - mean.get(feature, 0.0)) / max(float(std.get(feature, 1.0)), 1e-12)
        output = output + config.stress_weights[feature] * z.fillna(0.0)
    return output


def _macro_multiplier(
    frame: pd.DataFrame,
    overlay: dict[str, Any],
    config: LGDForwardLookingConfig,
) -> pd.Series:
    stress = _stress_index(frame, config, fitted=overlay)
    multiplier = np.exp(overlay["validation_intercept"] + overlay["slope"] * stress)
    lower = config.overlay_floor if config.overlay_floor is not None else 0.0
    upper = config.overlay_cap if config.overlay_cap is not None else np.inf
    return pd.Series(np.clip(multiplier, lower, upper), index=frame.index)


def _clip_lgd(values: pd.Series) -> pd.Series:
    return values.clip(lower=0.0, upper=1.0)


def _rating_structural_base(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    calibration = frame.get("expected_lgd_calibration_factor", pd.Series(1.0, index=frame.index))
    frame["calibrated_predicted_cure_lgd"] = (
        frame["predicted_cure_lgd"] * calibration
    ).clip(0, 1)
    frame["calibrated_predicted_non_cure_lgd"] = (
        frame["predicted_non_cure_lgd"] * calibration
    ).clip(0, 1)
    return (
        frame.groupby("rating_at_default", observed=True)
        .agg(
            lgd_base_structural=("predicted_lgd", "mean"),
            predicted_cure_probability=("predicted_cure_probability", "mean"),
            predicted_cure_lgd=("calibrated_predicted_cure_lgd", "mean"),
            predicted_non_cure_lgd=("calibrated_predicted_non_cure_lgd", "mean"),
            observed_lgd=("realized_lgd_model_target", "mean"),
            rows=("loan_id", "size"),
        )
        .reset_index()
        .rename(columns={"rating_at_default": "rating"})
    )


def _scenario_lgd_by_rating(
    rating_base: pd.DataFrame,
    scenario_macro: pd.DataFrame,
    overlay: dict[str, Any],
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    rows = []
    for macro_row in scenario_macro.itertuples(index=False):
        macro_frame = pd.DataFrame([macro_row._asdict()])
        multiplier = float(_macro_multiplier(macro_frame, overlay, config).iloc[0])
        for rating in rating_base.itertuples(index=False):
            cure_lgd = min(1.0, max(0.0, rating.predicted_cure_lgd * multiplier))
            non_cure_lgd = min(1.0, max(0.0, rating.predicted_non_cure_lgd * multiplier))
            lgd = (
                rating.predicted_cure_probability * cure_lgd
                + (1 - rating.predicted_cure_probability) * non_cure_lgd
            )
            rows.append(
                {
                    "scenario": macro_row.scenario,
                    "date": macro_row.date,
                    "rating": rating.rating,
                    "lgd_base_structural": rating.lgd_base_structural,
                    "predicted_cure_probability": rating.predicted_cure_probability,
                    "scenario_cure_lgd": cure_lgd,
                    "scenario_non_cure_lgd": non_cure_lgd,
                    "macro_overlay": multiplier,
                    "lgd_scenario": lgd,
                    "scenario_weight": macro_row.scenario_weight,
                    "weighted_lgd": lgd * macro_row.scenario_weight,
                }
            )
    return pd.DataFrame(rows)


def _scenario_lgd_from_stressed_ltv(
    episodes: pd.DataFrame,
    scenario_macro: pd.DataFrame,
    cure_model: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    scenario_macro = _add_cumulative_hpi_change(scenario_macro)
    features = cure_model["features"]
    factor = cure_model["calibrator"].get("factor", 1.0)
    for macro_row in scenario_macro.itertuples(index=False):
        stressed = episodes.copy()
        hpi_change = float(macro_row.cumulative_hpi_change)
        denominator = max(1 + hpi_change, 0.05)
        stressed["stressed_ltv"] = (
            stressed["estimated_loan_to_value_at_default"] / denominator
        ).clip(0, 300)
        if "estimated_loan_to_value_at_default" in features:
            stressed["estimated_loan_to_value_at_default"] = stressed["stressed_ltv"]
        cure_lgd = pd.Series(
            np.clip(cure_model["model"].predict(stressed[features]) * factor, 0, 1),
            index=stressed.index,
        )
        branch_elgd = (
            stressed["predicted_cure_probability"] * cure_lgd
            + (1 - stressed["predicted_cure_probability"])
            * stressed["predicted_non_cure_lgd"]
        )
        calibration = stressed.get(
            "expected_lgd_calibration_factor",
            pd.Series(1.0, index=stressed.index),
        )
        stressed["scenario_cure_lgd"] = cure_lgd
        stressed["scenario_non_cure_lgd"] = stressed["predicted_non_cure_lgd"]
        stressed["lgd_scenario"] = np.clip(branch_elgd * calibration, 0, 1)
        for rating, group in stressed.groupby("rating_at_default", observed=True):
            rows.append(
                {
                    "scenario": macro_row.scenario,
                    "date": macro_row.date,
                    "rating": rating,
                    "lgd_base_structural": group["predicted_lgd"].mean(),
                    "predicted_cure_probability": group["predicted_cure_probability"].mean(),
                    "scenario_cure_lgd": group["scenario_cure_lgd"].mean(),
                    "scenario_non_cure_lgd": group["scenario_non_cure_lgd"].mean(),
                    "macro_overlay": 1.0,
                    "cumulative_hpi_change": hpi_change,
                    "mean_stressed_ltv": group["stressed_ltv"].mean(),
                    "lgd_scenario": group["lgd_scenario"].mean(),
                    "scenario_weight": macro_row.scenario_weight,
                    "weighted_lgd": group["lgd_scenario"].mean() * macro_row.scenario_weight,
                }
            )
    return pd.DataFrame(rows)


def _add_cumulative_hpi_change(scenario_macro: pd.DataFrame) -> pd.DataFrame:
    frame = scenario_macro.sort_values(["scenario", "date"]).copy()
    quarterly_growth = frame["house_price_index_yoy"].fillna(0) / 100 / 4
    frame["cumulative_hpi_change"] = (
        (1 + quarterly_growth).groupby(frame["scenario"]).cumprod() - 1
    )
    return frame


def _weighted_scenario_lgd(scenario_lgd: pd.DataFrame) -> pd.DataFrame:
    return (
        scenario_lgd.groupby(["date", "rating"], observed=True)
        .agg(
            probability_weighted_lgd=("weighted_lgd", "sum"),
            scenario_weight_sum=("scenario_weight", "sum"),
            lgd_base_structural=("lgd_base_structural", "first"),
        )
        .reset_index()
    )


def _scenario_sensitivity(
    episodes: pd.DataFrame,
    scenario_macro: pd.DataFrame,
    overlay: dict[str, Any],
    config: LGDForwardLookingConfig,
    cure_model: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows = []
    base_mean = episodes["predicted_lgd"].mean()
    for hpi in config.sensitivity.hpi_shocks:
        for unemployment in config.sensitivity.unemployment_shocks:
            shocked = scenario_macro.copy()
            shocked["house_price_index_yoy"] = shocked["house_price_index_yoy"] + hpi
            shocked["unemployment_rate"] = shocked["unemployment_rate"] + unemployment
            if cure_model and cure_model["model"] is not None:
                scenario = _scenario_lgd_from_stressed_ltv(episodes, shocked, cure_model)
            else:
                rating_base = _rating_structural_base(episodes)
                scenario = _scenario_lgd_by_rating(rating_base, shocked, overlay, config)
            rows.append(
                {
                    "hpi_shock": hpi,
                    "unemployment_shock": unemployment,
                    "mean_probability_weighted_lgd": _weighted_scenario_lgd(scenario)[
                        "probability_weighted_lgd"
                    ].mean(),
                    "mean_structural_lgd": base_mean,
                }
            )
    return pd.DataFrame(rows)


def _relationship_diagnostics(
    historical: pd.DataFrame,
    config: LGDForwardLookingConfig,
) -> pd.DataFrame:
    targets = ["cure_rate", "cure_lgd", "non_cure_lgd", "realized_lgd"]
    rows = []
    for feature in config.macro_features + ["average_ltv", "r5_share", "delinquency_default_share"]:
        for target in targets:
            subset = historical[[feature, target]].dropna()
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": float(subset[feature].corr(subset[target]))
                    if len(subset) > 1
                    else np.nan,
                    "rows": len(subset),
                }
            )
    return pd.DataFrame(rows)


def _coefficient_table(overlay: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"term": "INTERCEPT_VALIDATION_CALIBRATED", "coefficient": overlay["validation_intercept"]},
        {"term": "stress_index", "coefficient": overlay["slope"]},
    ]
    for feature, weight in overlay["stress_weights"].items():
        rows.append({"term": f"stress_weight_{feature}", "coefficient": weight})
    return pd.DataFrame(rows)


def _historical_backtest(
    scored: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    groups = group_columns or ["split"]
    rows = []
    for keys, group in scored.groupby(groups, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys, strict=False))
        actual = group["realized_lgd_model_target"]
        base = group["predicted_lgd"]
        adjusted = group["lgd_forward_adjusted"]
        row.update(
            {
                "rows": len(group),
                "actual_mean_lgd": actual.mean(),
                "lgd_v1_2_predicted_mean": base.mean(),
                "lgd_v1_2_oe": actual.mean() / base.mean(),
                "forward_adjusted_predicted_mean": adjusted.mean(),
                "forward_adjusted_oe": actual.mean() / adjusted.mean(),
                "lgd_v1_2_mae": mean_absolute_error(actual, base),
                "forward_adjusted_mae": mean_absolute_error(actual, adjusted),
                "lgd_v1_2_rmse": mean_squared_error(actual, base) ** 0.5,
                "forward_adjusted_rmse": mean_squared_error(actual, adjusted) ** 0.5,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _oot_cure_lgd_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", observed=True):
        cured = group[group["cured_flag"]]
        rows.append(
            {
                "split": split,
                "rows": len(group),
                "cure_rows": len(cured),
                "cure_rate": group["cured_flag"].mean(),
                "realized_cure_lgd": cured["realized_lgd_model_target"].mean(),
                "predicted_cure_lgd": cured["predicted_cure_lgd"].mean(),
                "average_hpi_yoy": group["house_price_index_yoy"].mean(),
                "average_ltv": group["estimated_loan_to_value_at_default"].mean(),
                "average_systematic_factor": group["systematic_factor"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _scenario_ordering_diagnostics(scenario_lgd: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        scenario_lgd.groupby(["date", "rating", "scenario"], observed=True)["lgd_scenario"]
        .mean()
        .unstack("scenario")
        .reset_index()
    )
    if {"DOWNSIDE", "BASE", "UPSIDE"}.issubset(pivot.columns):
        pivot["ordered"] = (pivot["DOWNSIDE"] >= pivot["BASE"]) & (pivot["BASE"] >= pivot["UPSIDE"])
    else:
        pivot["ordered"] = False
    return pivot


def _load_overlay_model(repo_root: Path, config: LGDForwardLookingConfig) -> dict[str, Any]:
    model_root = repo_root / config.output.model_root
    candidates = sorted(model_root.glob("*/lgd_forward_overlay.pkl"))
    if not candidates:
        msg = "No persisted LGD forward-looking overlay model found"
        raise FileNotFoundError(msg)
    with candidates[-1].open("rb") as stream:
        payload = pickle.load(stream)
    return payload["overlay"]


def _load_cure_lgd_model(repo_root: Path, lgd_run_id: str) -> dict[str, Any]:
    path = repo_root / "models" / "lgd" / lgd_run_id / "lgd_models.pkl"
    if not path.exists():
        return {"model": None, "features": [], "calibrator": {"factor": 1.0}}
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    cure_lgd = payload["models"].get("cure_lgd", {})
    return {
        "model": cure_lgd.get("model"),
        "features": cure_lgd.get("features", []),
        "calibrator": cure_lgd.get("calibrator", {"factor": 1.0}),
    }


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
