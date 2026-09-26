"""Metrics for logistic scorecard experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics


def weighted_ks(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray) -> float:
    """Calculate weighted Kolmogorov-Smirnov statistic."""
    frame = pd.DataFrame({"y": y_true, "score": y_score, "weight": sample_weight})
    frame = frame.sort_values("score", ascending=False)
    total_bad = frame.loc[frame["y"] == 1, "weight"].sum()
    total_good = frame.loc[frame["y"] == 0, "weight"].sum()
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = (frame["weight"] * (frame["y"] == 1)).cumsum() / total_bad
    cum_good = (frame["weight"] * (frame["y"] == 0)).cumsum() / total_good
    return float((cum_bad - cum_good).abs().max())


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    """Return weighted binary classification metrics."""
    auc = metrics.roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    pr_auc = metrics.average_precision_score(y_true, y_prob, sample_weight=sample_weight)
    brier = metrics.brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
    loss = metrics.log_loss(y_true, np.clip(y_prob, 1e-8, 1 - 1e-8), sample_weight=sample_weight)
    observed = float(np.average(y_true, weights=sample_weight))
    predicted = float(np.average(y_prob, weights=sample_weight))
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": weighted_ks(y_true, y_prob, sample_weight),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "log_loss": float(loss),
        "observed_bad_rate": observed,
        "predicted_bad_rate": predicted,
    }


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    """Calculate population stability index using expected quantile breaks."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.maximum(expected_counts / expected_counts.sum(), 1e-6)
    actual_pct = np.maximum(actual_counts / actual_counts.sum(), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def score_bands(scores: pd.Series, train_scores: pd.Series, band_count: int) -> pd.Series:
    """Assign score bands from TRAIN quantile cut points."""
    edges = np.unique(train_scores.quantile(np.linspace(0, 1, band_count + 1)).to_numpy())
    if len(edges) < 2:
        return pd.Series("B01", index=scores.index)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"B{index:02d}" for index in range(1, len(edges))]
    return pd.cut(scores, bins=edges, labels=labels, include_lowest=True).astype(str)
