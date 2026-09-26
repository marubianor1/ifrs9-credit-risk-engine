"""Logistic scorecard experiment runner."""

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
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ifrs9.development.factory import run_development_sample_build
from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.feature_registry import load_default_registry
from ifrs9.mart.views import register_gold_views
from ifrs9.models.scorecard.config import ScorecardConfig, load_scorecard_config
from ifrs9.models.scorecard.metrics import binary_metrics, psi, score_bands, weighted_ks
from ifrs9.models.scorecard.woe import WoeBinner


@dataclass(frozen=True)
class ScorecardRunResult:
    """Result of a scorecard experiment run."""

    run_id: str
    population: str
    artifact_path: str
    model_path: str
    processing_time_seconds: float


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


def _feature_use(population: str) -> str:
    return "application_scoring" if population == "application" else "behavioural_scoring"


def _apply_overrides(
    config: ScorecardConfig,
    *,
    population: str | None,
    snapshot_frequency: str | None,
    sampling_strategy: str | None,
) -> ScorecardConfig:
    updates: dict[str, object] = {}
    if population:
        updates["population"] = population
    if snapshot_frequency:
        updates["snapshot_frequency"] = snapshot_frequency
    if sampling_strategy:
        updates["sampling_strategy"] = sampling_strategy
    return config.model_copy(update=updates)


def _load_frame(repo_root: Path, features: list[str]) -> pd.DataFrame:
    feature_sql = ", ".join(f"a.{feature}" for feature in features)
    development_path = repo_root / "data" / "gold" / "freddie" / "development"
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        con.execute(
            f"""
            CREATE OR REPLACE VIEW development_sample AS
            SELECT * FROM read_parquet('{development_path.as_posix()}/*.parquet')
            """
        )
        query = f"""
            SELECT
                d.loan_id,
                d.as_of_date,
                d.split,
                d.observation_year,
                d.default_next_12m AS target,
                d.sample_weight,
                d.sampling_probability,
                a.vintage_year,
                {feature_sql}
            FROM development_sample d
            JOIN gold_loan_month_analytical a
              ON d.loan_id = a.loan_id
             AND d.as_of_date = a.as_of_date
            WHERE d.development_eligible_12m
              AND d.sample_selected
              AND d.split IN ('TRAIN', 'VALIDATION', 'OOT')
        """
        return con.execute(query).fetchdf()


def _train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    weights: pd.Series,
) -> LogisticRegression:
    model = LogisticRegression(max_iter=300, solver="lbfgs", random_state=0)
    model.fit(train_x, train_y, sample_weight=weights)
    check_is_fitted(model)
    return model


def _score_from_pd(probability: np.ndarray, config: ScorecardConfig) -> np.ndarray:
    factor = config.score_scaling.pdo / np.log(2)
    odds = (1 - probability) / np.clip(probability, 1e-8, 1 - 1e-8)
    return config.score_scaling.base_score + factor * (
        np.log(odds) - np.log(config.score_scaling.base_odds_good_to_bad)
    )


def _calibration(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return {"calibration_intercept": 0.0, "calibration_slope": 0.0}
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(logit(clipped).reshape(-1, 1), y_true, sample_weight=weights)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def _predictions(
    frame: pd.DataFrame,
    features: list[str],
    binner: WoeBinner,
    model: LogisticRegression,
    config: ScorecardConfig,
) -> pd.DataFrame:
    woe_x = binner.transform(frame[features])
    probability = model.predict_proba(woe_x)[:, 1]
    scored = frame[["loan_id", "as_of_date", "split", "observation_year", "vintage_year"]].copy()
    scored["target"] = frame["target"].astype(int)
    scored["sample_weight"] = frame["sample_weight"].astype(float)
    scored["predicted_pd_raw"] = probability
    scored["score"] = _score_from_pd(probability, config)
    scored["linear_predictor"] = model.decision_function(woe_x)
    return scored


def _metric_rows(scored: pd.DataFrame, config: ScorecardConfig) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=True):
        y = group["target"].to_numpy()
        p = group["predicted_pd_raw"].to_numpy()
        w = group["sample_weight"].to_numpy()
        row = {"split": split, **binary_metrics(y, p, w), **_calibration(y, p, w)}
        cutoff = config.metrics.confusion_cutoff
        row["confusion_cutoff"] = cutoff
        row["true_positive"] = float(np.sum(w[(p >= cutoff) & (y == 1)]))
        row["false_positive"] = float(np.sum(w[(p >= cutoff) & (y == 0)]))
        row["true_negative"] = float(np.sum(w[(p < cutoff) & (y == 0)]))
        row["false_negative"] = float(np.sum(w[(p < cutoff) & (y == 1)]))
        rows.append(row)
    return pd.DataFrame(rows)


def _decile_table(scored: pd.DataFrame, config: ScorecardConfig) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=True):
        ranked = group.copy()
        ranked["decile"] = pd.qcut(
            ranked["predicted_pd_raw"].rank(method="first"),
            config.metrics.deciles,
            labels=False,
            duplicates="drop",
        ) + 1
        for decile, bucket in ranked.groupby("decile"):
            rows.append(
                {
                    "split": split,
                    "decile": int(decile),
                    "rows": len(bucket),
                    "weighted_bad_rate": np.average(
                        bucket["target"],
                        weights=bucket["sample_weight"],
                    ),
                    "weighted_predicted_pd": np.average(
                        bucket["predicted_pd_raw"],
                        weights=bucket["sample_weight"],
                    ),
                    "min_score": bucket["score"].min(),
                    "max_score": bucket["score"].max(),
                }
            )
    return pd.DataFrame(rows)


def _curve_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roc_rows = []
    calibration_rows = []
    lift_rows = []
    for split, group in scored.groupby("split", sort=True):
        y = group["target"].to_numpy()
        p = group["predicted_pd_raw"].to_numpy()
        w = group["sample_weight"].to_numpy()
        fpr, tpr, thresholds = _weighted_roc_curve(y, p, w)
        roc_rows.extend(
            {"split": split, "fpr": fp, "tpr": tp, "threshold": threshold}
            for fp, tp, threshold in zip(fpr, tpr, thresholds, strict=False)
        )
        bins = pd.qcut(group["predicted_pd_raw"].rank(method="first"), 10, labels=False) + 1
        tmp = group.assign(calibration_bin=bins)
        overall_bad = np.average(group["target"], weights=group["sample_weight"])
        cumulative_bad = 0.0
        cumulative_weight = 0.0
        for bin_id, bucket in tmp.sort_values("predicted_pd_raw", ascending=False).groupby(
            "calibration_bin",
            sort=True,
        ):
            bad = float((bucket["target"] * bucket["sample_weight"]).sum())
            weight = float(bucket["sample_weight"].sum())
            cumulative_bad += bad
            cumulative_weight += weight
            calibration_rows.append(
                {
                    "split": split,
                    "bin": int(bin_id),
                    "observed_bad_rate": bad / weight if weight else 0.0,
                    "predicted_bad_rate": np.average(
                        bucket["predicted_pd_raw"],
                        weights=bucket["sample_weight"],
                    ),
                }
            )
            lift_rows.append(
                {
                    "split": split,
                    "bin": int(bin_id),
                    "cumulative_capture_rate": cumulative_bad
                    / float((group["target"] * group["sample_weight"]).sum()),
                    "lift": (cumulative_bad / cumulative_weight) / overall_bad,
                }
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(calibration_rows), pd.DataFrame(lift_rows)


def _weighted_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob)
    y = y_true[order]
    p = y_prob[order]
    w = sample_weight[order]
    tp = np.cumsum(w * (y == 1))
    fp = np.cumsum(w * (y == 0))
    total_pos = tp[-1] if len(tp) else 0
    total_neg = fp[-1] if len(fp) else 0
    if total_pos == 0 or total_neg == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])
    return fp / total_neg, tp / total_pos, p


def _stability_tables(
    scored: pd.DataFrame,
    feature_frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = scored[scored["split"] == "TRAIN"]
    rows = []
    for split in ["VALIDATION", "OOT"]:
        group = scored[scored["split"] == split]
        rows.append(
            {
                "type": "score",
                "feature": "score",
                "comparison": f"TRAIN_TO_{split}",
                "psi": psi(train["score"].to_numpy(), group["score"].to_numpy()),
            }
        )
    for feature in features:
        train_values = feature_frame.loc[scored["split"] == "TRAIN", feature].to_numpy(dtype=float)
        for split in ["VALIDATION", "OOT"]:
            split_values = feature_frame.loc[
                scored["split"] == split,
                feature,
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "type": "feature",
                    "feature": feature,
                    "comparison": f"TRAIN_TO_{split}",
                    "psi": psi(train_values, split_values),
                }
            )
    yearly = []
    for year, group in scored.groupby("observation_year"):
        y = group["target"].to_numpy()
        p = group["predicted_pd_raw"].to_numpy()
        w = group["sample_weight"].to_numpy()
        yearly.append(
            {
                "observation_year": int(year),
                "rows": len(group),
                "bad_rate": np.average(y, weights=w),
                "predicted_bad_rate": np.average(p, weights=w),
                "gini": 2 * _safe_auc(y, p, w) - 1,
                "ks": weighted_ks(y, p, w),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob, sample_weight=weights))


def _coefficient_table(model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    rows = [
        {
            "feature": "INTERCEPT",
            "coefficient": float(model.intercept_[0]),
            "odds_ratio": float(np.exp(model.intercept_[0])),
            "standard_error": None,
            "p_value": None,
            "sign_expected": None,
            "sign_matches_woe_expectation": None,
        }
    ]
    for feature, coefficient in zip(features, model.coef_[0], strict=False):
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "odds_ratio": float(np.exp(coefficient)),
                "standard_error": None,
                "p_value": None,
                "sign_expected": "negative",
                "sign_matches_woe_expectation": bool(coefficient <= 0),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def run_scorecard(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    population: str | None = None,
    snapshot_frequency: str | None = None,
    sampling_strategy: str | None = None,
    run_id: str | None = None,
) -> ScorecardRunResult:
    """Run a logistic scorecard experiment."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = _apply_overrides(
        load_scorecard_config(root, config_path),
        population=population,
        snapshot_frequency=snapshot_frequency,
        sampling_strategy=sampling_strategy,
    )
    registry = load_default_registry(root)
    features = registry.get_features_for(_feature_use(config.population))
    if not features:
        msg = f"No approved features for population {config.population}"
        raise ValueError(msg)

    run_development_sample_build(
        force=True,
        repo_root=root,
        snapshot_frequency=config.snapshot_frequency,
        sampling_strategy=config.sampling_strategy,
        population=config.population,
    )
    frame = _load_frame(root, features)
    run_id = run_id or f"{config.population}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / config.run.artifact_root / run_id
    model_dir = root / config.run.model_root / run_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    artifact_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    train = frame[frame["split"] == "TRAIN"].copy()
    x_train = train[features]
    y_train = train["target"].astype(int)
    w_train = train["sample_weight"].astype(float)
    binner = WoeBinner(
        max_bins=config.binning.max_bins,
        min_bin_pct=config.binning.min_bin_pct,
        enforce_monotonic=config.binning.enforce_monotonic,
        rare_category_pct=config.binning.rare_category_pct,
        smoothing=config.binning.smoothing,
    ).fit(x_train, y_train, sample_weight=w_train)
    train_woe = binner.transform(x_train)
    model = _train_model(train_woe, y_train, w_train)

    scored = _predictions(frame, features, binner, model, config)
    train_scores = scored.loc[scored["split"] == "TRAIN", "score"]
    scored["score_band"] = score_bands(
        scored["score"],
        train_scores,
        config.score_scaling.band_count,
    )

    metrics_table = _metric_rows(scored, config)
    deciles = _decile_table(scored, config)
    roc, calibration, lift = _curve_tables(scored)
    binning = binner.bin_table()
    binning["suspicious_high_iv"] = (
        binning["feature"].map(binner.feature_iv_) > config.binning.suspicious_iv_threshold
    )
    iv_ranking = (
        pd.DataFrame(
            {
                "feature": list(binner.feature_iv_.keys()),
                "iv": list(binner.feature_iv_.values()),
            }
        )
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    coefficients = _coefficient_table(model, features)
    psi_table, yearly = _stability_tables(scored, binner.transform(frame[features]), features)
    bands = (
        scored.groupby(["split", "score_band"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "bad_rate": np.average(group["target"], weights=group["sample_weight"]),
                    "predicted_bad_rate": np.average(
                        group["predicted_pd_raw"],
                        weights=group["sample_weight"],
                    ),
                    "min_score": group["score"].min(),
                    "max_score": group["score"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    pd.DataFrame({"feature": features}).to_csv(artifact_dir / "features.csv", index=False)
    binning.to_csv(artifact_dir / "binning.csv", index=False)
    iv_ranking.to_csv(artifact_dir / "iv_ranking.csv", index=False)
    coefficients.to_csv(artifact_dir / "coefficients.csv", index=False)
    metrics_table.to_csv(artifact_dir / "metrics.csv", index=False)
    deciles.to_csv(artifact_dir / "deciles.csv", index=False)
    roc.to_csv(artifact_dir / "roc_curve.csv", index=False)
    calibration.to_csv(artifact_dir / "calibration_curve.csv", index=False)
    lift.to_csv(artifact_dir / "gains_lift.csv", index=False)
    psi_table.to_csv(artifact_dir / "stability_psi.csv", index=False)
    yearly.to_csv(artifact_dir / "yearly_performance.csv", index=False)
    bands.to_csv(artifact_dir / "score_bands.csv", index=False)
    scored[
        [
            "loan_id",
            "as_of_date",
            "split",
            "target",
            "predicted_pd_raw",
            "score",
            "score_band",
        ]
    ].to_parquet(artifact_dir / "scored_rows.parquet", index=False)
    with (model_dir / "model.pkl").open("wb") as stream:
        pickle.dump(
            {"model": model, "binner": binner, "features": features, "config": config},
            stream,
        )

    result = ScorecardRunResult(
        run_id=run_id,
        population=config.population,
        artifact_path=str(artifact_dir),
        model_path=str(model_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    _write_json(
        artifact_dir / "run.json",
        {
            "run_id": run_id,
            "model_type": config.model_type,
            "population": config.population,
            "config": config.model_dump(),
            "git_commit": _git_state(root),
            "features": features,
            "created_at": datetime.now(UTC).isoformat(),
            "result": asdict(result),
        },
    )
    return result


def load_run(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Load a persisted scorecard run manifest."""
    config = load_scorecard_config(repo_root)
    path = repo_root / config.run.artifact_root / run_id / "run.json"
    return json.loads(path.read_text())


def compare_runs(repo_root: Path, run_ids: list[str]) -> pd.DataFrame:
    """Compare persisted scorecard runs by top-level split metrics."""
    config = load_scorecard_config(repo_root)
    rows = []
    for run_id in run_ids:
        metrics_path = repo_root / config.run.artifact_root / run_id / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "run_id", run_id)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)
