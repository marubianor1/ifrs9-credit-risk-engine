"""Monitoring services for the Streamlit app."""

from __future__ import annotations

import yaml

from app.services.artifacts import artifact_path, read_csv


def load_monitoring_thresholds() -> dict:
    """Load transparent project monitoring thresholds."""
    with artifact_path("config", "monitoring.yaml").open() as stream:
        return yaml.safe_load(stream)["monitoring"]["thresholds"]


def status_band(value: float, threshold: dict, *, higher_is_better: bool) -> str:
    """Assign GREEN, AMBER, or RED using project thresholds."""
    if higher_is_better:
        if value >= threshold["green_min"]:
            return "GREEN"
        if value >= threshold["amber_min"]:
            return "AMBER"
        return "RED"
    if value <= threshold["green_max"]:
        return "GREEN"
    if value <= threshold["amber_max"]:
        return "AMBER"
    return "RED"


def oe_status(value: float, threshold: dict) -> str:
    """Assign status for O/E ratios."""
    if threshold["green_min"] <= value <= threshold["green_max"]:
        return "GREEN"
    if threshold["amber_min"] <= value <= threshold["amber_max"]:
        return "AMBER"
    return "RED"


def load_monitoring_artifacts() -> dict:
    """Load compact artifacts for monitoring dashboards."""
    return {
        "scorecard_metrics": read_csv(
            artifact_path("artifacts", "models", "scorecard", "behavioural_qe_v1", "metrics.csv")
        ),
        "scorecard_psi": read_csv(
            artifact_path(
                "artifacts",
                "models",
                "scorecard",
                "behavioural_qe_v1",
                "stability_psi.csv",
            )
        ),
        "scorecard_yearly": read_csv(
            artifact_path(
                "artifacts",
                "models",
                "scorecard",
                "behavioural_qe_v1",
                "yearly_performance.csv",
            )
        ),
        "pd_calibration": read_csv(
            artifact_path("artifacts", "pd", "pd_behavioural_qe_v1", "calibration_metrics.csv")
        ),
        "pd_backtest": read_csv(
            artifact_path("artifacts", "pd", "pd_behavioural_qe_v1", "backtesting_split.csv")
        ),
        "pd_rating": read_csv(
            artifact_path("artifacts", "pd", "pd_behavioural_qe_v1", "backtesting_rating_year.csv")
        ),
        "lgd_backtest": read_csv(
            artifact_path("artifacts", "lgd", "lgd_v1_2", "backtesting_split.csv")
        ),
        "lgd_components": read_csv(
            artifact_path("artifacts", "lgd", "lgd_v1_2", "component_decomposition.csv")
        ),
        "ead_backtest": read_csv(
            artifact_path("artifacts", "ead", "ead_v1", "backtest_by_split.csv")
        ),
    }
