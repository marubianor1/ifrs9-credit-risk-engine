"""Scorecard artifact services for the Streamlit app."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from app.services.artifacts import (
    artifact_path,
    discover_scorecard_runs,
    read_csv,
    read_json,
    repo_root,
)

DEFAULT_SCORECARD_RUNS = ["behavioural_qe_v1", "application_qe_v1"]


def available_scorecard_runs() -> list[str]:
    """Return scorecard runs with preferred portfolio ordering."""
    runs = discover_scorecard_runs()
    ordered = [run for run in DEFAULT_SCORECARD_RUNS if run in runs]
    return ordered + [run for run in runs if run not in ordered]


def load_scorecard_artifacts(run_id: str) -> dict[str, pd.DataFrame | dict]:
    """Load scorecard outputs for a selected run."""
    root = artifact_path("artifacts", "models", "scorecard", run_id)
    return {
        "run": read_json(root / "run.json"),
        "features": read_csv(root / "features.csv"),
        "iv": read_csv(root / "iv_ranking.csv"),
        "coefficients": read_csv(root / "coefficients.csv"),
        "metrics": read_csv(root / "metrics.csv"),
        "calibration": read_csv(root / "calibration_curve.csv"),
        "psi": read_csv(root / "stability_psi.csv"),
        "yearly": read_csv(root / "yearly_performance.csv"),
        "bands": read_csv(root / "score_bands.csv"),
        "deciles": read_csv(root / "deciles.csv"),
    }


def run_scorecard_from_controls(
    *,
    population: str,
    snapshot_frequency: str,
    sampling_strategy: str,
) -> str:
    """Call the existing scorecard backend with UI-supported controls."""
    from ifrs9.models.scorecard.runner import run_scorecard

    run_id = f"{population}_ui_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    result = run_scorecard(
        repo_root=repo_root(),
        population=population,
        snapshot_frequency=snapshot_frequency,
        sampling_strategy=sampling_strategy,
        run_id=run_id,
    )
    return result.run_id
