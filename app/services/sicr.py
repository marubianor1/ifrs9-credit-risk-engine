"""SICR and staging services for the Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.artifacts import artifact_path, read_csv, read_json, read_parquet, repo_root
from ifrs9.sicr.config import load_staging_config
from ifrs9.sicr.framework import (
    simulate_sicr_thresholds,
    simulate_threshold_stage_distribution,
)


def load_sicr_artifacts(run_id: str = "sicr_v1_1") -> dict[str, pd.DataFrame | dict]:
    """Load staging artifacts."""
    root = artifact_path("artifacts", "sicr", run_id)
    return {
        "run": read_json(root / "run.json"),
        "stage_distribution": read_csv(root / "stage_distribution.csv"),
        "trigger_distribution": read_csv(root / "trigger_distribution.csv"),
        "trigger_exclusivity": read_csv(root / "sicr_trigger_exclusivity.csv"),
        "migrations": read_csv(root / "stage_migrations.csv"),
        "stage3_audit": read_csv(root / "stage3_state_audit.csv"),
        "reference_lag": read_csv(root / "reference_lag_profile.csv"),
        "reference_pd": read_csv(root / "reference_pd_diagnostics.csv"),
        "threshold_sensitivity": read_csv(root / "threshold_sensitivity.csv"),
    }


def load_staging_dataset() -> pd.DataFrame:
    """Load compact persisted staging data."""
    return read_parquet(artifact_path("data", "gold", "freddie", "staging", "part-staging.parquet"))


def staging_overrides_from_controls(
    *,
    relative_pd: float,
    absolute_pd: float | None,
    rating_downgrade: int,
    dpd_backstop: int,
    cure_probation: int,
) -> dict[str, Any]:
    """Convert UI controls to staging simulation overrides."""
    _ = cure_probation
    return {
        "relative_pd_increase": [relative_pd],
        "absolute_pd_increase": absolute_pd,
        "rating_downgrade_notches": [rating_downgrade],
        "dpd_backstop_months": [dpd_backstop],
    }


def run_staging_simulation(overrides: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run existing staging threshold simulation on persisted staging output."""
    config = load_staging_config(repo_root())
    staged = load_staging_dataset()
    prepared = staged.assign(
        stage3_flag=staged["stage"].eq(3),
        low_credit_risk_exemption=False,
    )
    if "delinquency_months" not in prepared.columns:
        prepared = prepared.assign(delinquency_months=0)
    sensitivity = simulate_sicr_thresholds(config, prepared, overrides=overrides)
    single_overrides = {
        key: value[0] if isinstance(value, list) else value for key, value in overrides.items()
    }
    distribution = simulate_threshold_stage_distribution(config, prepared, single_overrides)
    return {"sensitivity": sensitivity, "stage_distribution": distribution}
