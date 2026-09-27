"""Calibrated PD, rating scale, TTC, and lifetime term-structure framework."""

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.models.scorecard.metrics import binary_metrics
from ifrs9.pd.config import PDFrameworkConfig, load_pd_config


@dataclass(frozen=True)
class PDRunResult:
    """Result of a PD framework run."""

    run_id: str
    scorecard_run_id: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


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


def _scorecard_artifact_dir(repo_root: Path, run_id: str) -> Path:
    return repo_root / "artifacts" / "models" / "scorecard" / run_id


def _load_scored_rows(repo_root: Path, scorecard_run_id: str) -> pd.DataFrame:
    path = _scorecard_artifact_dir(repo_root, scorecard_run_id) / "scored_rows.parquet"
    if not path.exists():
        msg = f"Scorecard scored rows not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(path)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date
    if "sample_weight" not in frame.columns:
        frame["sample_weight"] = 1.0
    return frame


def _fit_calibration(
    frame: pd.DataFrame,
    config: PDFrameworkConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    raw_column = config.calibration.raw_pd_column
    raw = np.clip(frame[raw_column].to_numpy(), 1e-6, 1 - 1e-6)
    y = frame["target"].to_numpy(dtype=int)
    weights = frame["sample_weight"].to_numpy(dtype=float)
    validation = frame["split"] == config.calibration.fit_split
    if not validation.any():
        msg = f"Calibration split has no rows: {config.calibration.fit_split}"
        raise ValueError(msg)
    method = config.calibration.method
    if method == "none":
        return {"method": "none", "raw_pd_column": raw_column}, raw
    if method == "logistic_recalibration":
        model = LogisticRegression(max_iter=300, solver="lbfgs")
        model.fit(
            logit(raw[validation]).reshape(-1, 1),
            y[validation],
            sample_weight=weights[validation],
        )
        coefficient = float(model.coef_[0][0])
        coefficient_was_clipped = coefficient < 0
        if coefficient < 0:
            coefficient = 0.0
        calibrated = expit(float(model.intercept_[0]) + coefficient * logit(raw))
        return {
            "method": method,
            "raw_pd_column": raw_column,
            "intercept": float(model.intercept_[0]),
            "coefficient": coefficient,
            "coefficient_was_clipped": coefficient_was_clipped,
            "fit_split": config.calibration.fit_split,
            "fit_rows": int(validation.sum()),
        }, calibrated
    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(raw[validation], y[validation], sample_weight=weights[validation])
    return {
        "method": method,
        "raw_pd_column": raw_column,
        "fit_split": config.calibration.fit_split,
        "fit_rows": int(validation.sum()),
    }, model.predict(raw)


def _calibration_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split", sort=True):
        y = group["target"].to_numpy(dtype=int)
        w = group["sample_weight"].to_numpy(dtype=float)
        for column, label in [("pd_raw", "raw"), ("pd_calibrated_12m", "calibrated")]:
            p = np.clip(group[column].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            metrics = binary_metrics(y, p, w)
            metrics["ece"] = _expected_calibration_error(y, p, w)
            metrics["split"] = split
            metrics["pd_type"] = label
            rows.append(metrics)
    return pd.DataFrame(rows)


def _expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = float(w.sum())
    if total == 0:
        return 0.0
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= left) & (p <= right if right == 1 else p < right)
        if not mask.any():
            continue
        weight = float(w[mask].sum())
        observed = float(np.average(y[mask], weights=w[mask]))
        predicted = float(np.average(p[mask], weights=w[mask]))
        error += weight / total * abs(observed - predicted)
    return error


def _rating_boundaries(frame: pd.DataFrame, config: PDFrameworkConfig) -> pd.DataFrame:
    train = frame[frame["split"] == "TRAIN"].copy()
    labels = config.rating_master_scale.labels
    quantiles = np.linspace(0, 1, config.rating_master_scale.grade_count + 1)
    edges = np.unique(train["pd_calibrated_12m"].quantile(quantiles).to_numpy())
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0] = 0.0
    edges[-1] = 1.0
    return pd.DataFrame(
        {
            "rating": labels[: len(edges) - 1],
            "lower_pd": edges[:-1],
            "upper_pd": edges[1:],
        }
    )


def _assign_ratings(frame: pd.DataFrame, boundaries: pd.DataFrame) -> pd.Series:
    bins = boundaries["lower_pd"].to_list() + [float(boundaries["upper_pd"].iloc[-1])]
    labels = boundaries["rating"].to_list()
    return pd.cut(
        frame["pd_calibrated_12m"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)


def _rating_summary(frame: pd.DataFrame, config: PDFrameworkConfig) -> pd.DataFrame:
    total_population = len(frame)
    summary = (
        frame.groupby("rating", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "population": len(group),
                    "defaults": float((group["target"] * group["sample_weight"]).sum()),
                    "observed_bad_rate": np.average(
                        group["target"],
                        weights=group["sample_weight"],
                    ),
                    "mean_pd": np.average(
                        group["pd_calibrated_12m"],
                        weights=group["sample_weight"],
                    ),
                    "min_pd": group["pd_calibrated_12m"].min(),
                    "max_pd": group["pd_calibrated_12m"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    minimum_population = total_population * config.rating_master_scale.minimum_population_pct
    summary["minimum_population_threshold"] = minimum_population
    summary["minimum_defaults_threshold"] = config.rating_master_scale.minimum_defaults
    summary["passes_minimum_population"] = summary["population"] >= minimum_population
    summary["passes_minimum_defaults"] = (
        summary["defaults"] >= config.rating_master_scale.minimum_defaults
    )
    return summary.sort_values("mean_pd").reset_index(drop=True)


def _ttc_tables(
    frame: pd.DataFrame,
    config: PDFrameworkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        (pd.to_datetime(frame["as_of_date"]) >= pd.Timestamp(config.ttc.anchor_start))
        & (pd.to_datetime(frame["as_of_date"]) <= pd.Timestamp(config.ttc.anchor_end))
    )
    anchor = frame[mask]
    if anchor.empty:
        msg = "TTC anchor window has no observable 12-month outcomes."
        raise ValueError(msg)
    portfolio = pd.DataFrame(
        [
            {
                "anchor_start": config.ttc.anchor_start,
                "anchor_end": config.ttc.anchor_end,
                "portfolio_ttc_pd": np.average(anchor["target"], weights=anchor["sample_weight"]),
                "observations": len(anchor),
                "weighting": config.ttc.weighting,
            }
        ]
    )
    rating = (
        anchor.groupby("rating", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "rating_ttc_pd": np.average(group["target"], weights=group["sample_weight"]),
                    "observations": len(group),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return portfolio, rating


def _pit_ttc_diagnostics(frame: pd.DataFrame, portfolio_ttc: float) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby(pd.to_datetime(frame["as_of_date"]).dt.year):
        observed = np.average(group["target"], weights=group["sample_weight"])
        predicted = np.average(group["pd_calibrated_12m"], weights=group["sample_weight"])
        rows.append(
            {
                "observation_year": int(year),
                "observed_12m_default_rate": observed,
                "mean_calibrated_pd": predicted,
                "ttc_pd": portfolio_ttc,
                "pit_to_ttc_ratio": predicted / portfolio_ttc if portfolio_ttc else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_event_foundation(repo_root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    target_path = repo_root / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
    keys = frame[["loan_id", "as_of_date", "rating"]].copy()
    with duckdb.connect(database=":memory:") as con:
        con.register("keys", keys)
        return con.execute(
            f"""
            SELECT k.loan_id, k.as_of_date, k.rating,
                   t.months_until_event, t.event_default,
                   t.event_prepayment_or_exit, t.event_censored, t.event_type
            FROM keys k
            JOIN read_parquet('{target_path.as_posix()}/*.parquet') t
              ON k.loan_id = t.loan_id
             AND k.as_of_date = t.as_of_date
            """
        ).fetchdf()


def _lifetime_curves(
    events: pd.DataFrame,
    frame: pd.DataFrame,
    config: PDFrameworkConfig,
) -> pd.DataFrame:
    rating_pd = (
        frame.groupby("rating", observed=True)
        .apply(
            lambda group: np.average(
                group["pd_calibrated_12m"],
                weights=group["sample_weight"],
            ),
            include_groups=False,
        )
        .to_dict()
    )
    rows = []
    max_horizon = config.lifetime.max_horizon_months
    for rating, group in events.groupby("rating", observed=True):
        survival = 1.0
        cumulative = 0.0
        rating_rows = []
        for month in range(1, max_horizon + 1):
            at_risk = group["months_until_event"] >= month
            defaults = (group["event_default"]) & (group["months_until_event"] == month)
            hazard = (
                (defaults.sum() + config.lifetime.smoothing)
                / (at_risk.sum() + config.lifetime.smoothing * 2)
                if at_risk.sum() > 0
                else 0.0
            )
            marginal = survival * hazard
            cumulative = min(1.0, cumulative + marginal)
            survival = max(0.0, survival - marginal)
            rating_rows.append(
                {
                    "rating": rating,
                    "month": month,
                    "conditional_pd": hazard,
                    "hazard": hazard,
                    "marginal_pd": marginal,
                    "cumulative_pd": cumulative,
                    "survival_probability": survival,
                }
            )
        if config.lifetime.reconcile_first_12m and rating_rows:
            cumulative_12 = rating_rows[11]["cumulative_pd"]
            target_12 = float(rating_pd.get(rating, cumulative_12))
            factor = _hazard_scale_factor(
                [row["conditional_pd"] for row in rating_rows[:12]],
                target_12,
            )
            survival = 1.0
            cumulative = 0.0
            for row in rating_rows:
                row["conditional_pd"] = min(1.0, row["conditional_pd"] * factor)
                row["hazard"] = row["conditional_pd"]
                row["marginal_pd"] = survival * row["conditional_pd"]
                cumulative = min(1.0, cumulative + row["marginal_pd"])
                survival = max(0.0, survival - row["marginal_pd"])
                row["cumulative_pd"] = cumulative
                row["survival_probability"] = survival
        rows.extend(rating_rows)
    return pd.DataFrame(rows)


def _hazard_scale_factor(hazards: list[float], target_cumulative: float) -> float:
    target = min(max(target_cumulative, 0.0), 1.0)
    if not hazards or target == 0:
        return 0.0
    if _scaled_cumulative(hazards, 1.0) == 0:
        return 1.0
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


def _transition_matrices(frame: pd.DataFrame, config: PDFrameworkConfig) -> dict[int, pd.DataFrame]:
    matrices = {}
    ordered = frame[["loan_id", "as_of_date", "rating", "target"]].copy()
    ordered["as_of_date"] = pd.to_datetime(ordered["as_of_date"])
    for months in config.transitions.frequencies_months:
        future = ordered.copy()
        future["as_of_date"] = future["as_of_date"] - pd.DateOffset(months=months)
        joined = ordered.merge(future, on=["loan_id", "as_of_date"], suffixes=("_from", "_to"))
        joined["to_state"] = np.where(
            joined["target_to"] == 1,
            config.transitions.default_state,
            joined["rating_to"],
        )
        matrix = (
            joined.groupby(["rating_from", "to_state"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        totals = matrix.groupby("rating_from")["count"].transform("sum")
        matrix["probability"] = matrix["count"] / totals
        default_rows = pd.DataFrame(
            [
                {
                    "rating_from": config.transitions.default_state,
                    "to_state": config.transitions.default_state,
                    "count": 0,
                    "probability": 1.0,
                }
            ]
        )
        matrices[months] = pd.concat([matrix, default_rows], ignore_index=True)
    return matrices


def _transition_summary(
    matrices: dict[int, pd.DataFrame],
    config: PDFrameworkConfig,
) -> pd.DataFrame:
    rows = []
    for months, matrix in matrices.items():
        non_default = matrix[matrix["rating_from"] != config.transitions.default_state].copy()
        for from_state, group in non_default.groupby("rating_from"):
            denominator = group["count"].sum()
            if denominator == 0:
                continue
            from_rank = _rating_rank(str(from_state))
            upgrade = 0.0
            downgrade = 0.0
            default = 0.0
            stable = 0.0
            for row in group.itertuples(index=False):
                to_state = str(row.to_state)
                if to_state == config.transitions.default_state:
                    default += float(row.count)
                    downgrade += float(row.count)
                    continue
                to_rank = _rating_rank(to_state)
                if to_rank < from_rank:
                    upgrade += float(row.count)
                elif to_rank > from_rank:
                    downgrade += float(row.count)
                else:
                    stable += float(row.count)
            rows.append(
                {
                    "months": months,
                    "rating_from": from_state,
                    "migration_rate": 1 - stable / denominator,
                    "upgrade_rate": upgrade / denominator,
                    "downgrade_rate": downgrade / denominator,
                    "default_transition_rate": default / denominator,
                    "transitions": int(denominator),
                }
            )
    return pd.DataFrame(rows)


def _rating_rank(rating: str) -> int:
    if rating.startswith("R") and rating[1:].isdigit():
        return int(rating[1:])
    return 10_000


def _backtesting(
    frame: pd.DataFrame,
    config: PDFrameworkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    years = pd.to_datetime(frame["as_of_date"]).dt.year
    for (rating, year), group in frame.groupby(["rating", years], observed=True):
        observed = float((group["target"] * group["sample_weight"]).sum())
        expected = float((group["pd_calibrated_12m"] * group["sample_weight"]).sum())
        n = float(group["sample_weight"].sum())
        p = expected / n if n else 0.0
        se = np.sqrt(max(p * (1 - p) / n, 0)) if n else 0.0
        rows.append(
            {
                "rating": rating,
                "observation_year": int(year),
                "observed_defaults": observed,
                "expected_defaults": expected,
                "oe_ratio": observed / expected if expected else np.nan,
                "binomial_lower": max(0.0, p - 1.96 * se),
                "binomial_upper": min(1.0, p + 1.96 * se),
                "brier": np.average(
                    (group["target"] - group["pd_calibrated_12m"]) ** 2,
                    weights=group["sample_weight"],
                ),
                "insufficient_defaults": (
                    observed < config.backtesting.insufficient_default_threshold
                ),
            }
        )
    portfolio = []
    for split, group in frame.groupby("split"):
        observed = float((group["target"] * group["sample_weight"]).sum())
        expected = float((group["pd_calibrated_12m"] * group["sample_weight"]).sum())
        portfolio.append(
            {
                "split": split,
                "observed_defaults": observed,
                "expected_defaults": expected,
                "oe_ratio": observed / expected if expected else np.nan,
                "brier": np.average(
                    (group["target"] - group["pd_calibrated_12m"]) ** 2,
                    weights=group["sample_weight"],
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(portfolio)


def run_pd_framework(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    scorecard_run_id: str | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> PDRunResult:
    """Run the calibrated PD framework."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_pd_config(root, config_path)
    parent_run_id = scorecard_run_id or config.primary_scorecard_run
    run_id = run_id or f"pd_{parent_run_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.run.artifact_root / run_id
    model_dir = root / config.run.model_root / run_id
    if not force and (artifact_dir.exists() or model_dir.exists()):
        msg = f"PD run already exists: {run_id}. Pass force=True or --force to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_scored_rows(root, parent_run_id)
    calibration_model, calibrated = _fit_calibration(frame, config)
    frame["pd_raw"] = frame[config.calibration.raw_pd_column]
    frame["pd_calibrated_12m"] = np.clip(calibrated, 1e-8, 1 - 1e-8)
    boundaries = _rating_boundaries(frame, config)
    frame["rating"] = _assign_ratings(frame, boundaries)
    rating_summary = _rating_summary(frame, config)
    portfolio_ttc, rating_ttc = _ttc_tables(frame, config)
    diagnostics = _pit_ttc_diagnostics(frame, float(portfolio_ttc["portfolio_ttc_pd"].iloc[0]))
    events = _load_event_foundation(root, frame)
    curves = _lifetime_curves(events, frame, config)
    transitions = _transition_matrices(frame, config)
    transition_summary = _transition_summary(transitions, config)
    by_rating_year, by_split = _backtesting(frame, config)
    metrics = _calibration_metrics(frame)
    curve = _calibration_curve(frame)

    frame[
        [
            "loan_id",
            "as_of_date",
            "split",
            "score",
            "pd_raw",
            "pd_calibrated_12m",
            "rating",
            "target",
        ]
    ].to_parquet(artifact_dir / "pd_predictions.parquet", index=False)
    metrics.to_csv(artifact_dir / "calibration_metrics.csv", index=False)
    curve.to_csv(artifact_dir / "calibration_curve.csv", index=False)
    boundaries.to_csv(artifact_dir / "rating_boundaries.csv", index=False)
    rating_summary.to_csv(artifact_dir / "rating_summary.csv", index=False)
    portfolio_ttc.to_csv(artifact_dir / "portfolio_ttc.csv", index=False)
    rating_ttc.to_csv(artifact_dir / "rating_ttc.csv", index=False)
    diagnostics.to_csv(artifact_dir / "pit_ttc_diagnostics.csv", index=False)
    curves.to_csv(artifact_dir / "lifetime_curves.csv", index=False)
    by_rating_year.to_csv(artifact_dir / "backtesting_rating_year.csv", index=False)
    by_split.to_csv(artifact_dir / "backtesting_split.csv", index=False)
    for months, matrix in transitions.items():
        matrix.to_csv(artifact_dir / f"transition_{months}m.csv", index=False)
    transition_summary.to_csv(artifact_dir / "transition_summary.csv", index=False)
    with (model_dir / "calibration_model.pkl").open("wb") as stream:
        pickle.dump(calibration_model, stream)

    result = PDRunResult(
        run_id=run_id,
        scorecard_run_id=parent_run_id,
        artifact_path=str(artifact_dir),
        model_path=str(model_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "scorecard_parent_run_id": parent_run_id,
        "config": config.model_dump(),
        "calibration_model": calibration_model,
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
    }
    (artifact_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return result


def _calibration_curve(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split"):
        ranked = group.copy()
        ranked["bucket"] = pd.qcut(
            ranked["pd_calibrated_12m"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        ) + 1
        for bucket, part in ranked.groupby("bucket"):
            rows.append(
                {
                    "split": split,
                    "bucket": int(bucket),
                    "observed_bad_rate": np.average(part["target"], weights=part["sample_weight"]),
                    "predicted_bad_rate": np.average(
                        part["pd_calibrated_12m"],
                        weights=part["sample_weight"],
                    ),
                    "rows": len(part),
                }
            )
    return pd.DataFrame(rows)


def load_pd_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted PD run manifest."""
    config = load_pd_config(repo_root)
    return json.loads((repo_root / config.run.artifact_root / run_id / "run.json").read_text())
