"""PD artifact services for the Streamlit app."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from app.services.artifacts import artifact_path, discover_runs, read_csv, read_json, repo_root


def available_pd_runs() -> list[str]:
    """Return available PD framework run IDs."""
    return discover_runs("pd")


def load_pd_artifacts(run_id: str) -> dict[str, pd.DataFrame | dict]:
    """Load PD framework outputs for a selected run."""
    root = artifact_path("artifacts", "pd", run_id)
    return {
        "run": read_json(root / "run.json"),
        "calibration_metrics": read_csv(root / "calibration_metrics.csv"),
        "calibration_curve": read_csv(root / "calibration_curve.csv"),
        "rating_summary": read_csv(root / "rating_summary.csv"),
        "rating_ttc": read_csv(root / "rating_ttc.csv"),
        "rating_boundaries": read_csv(root / "rating_boundaries.csv"),
        "pit_ttc": read_csv(root / "pit_ttc_diagnostics.csv"),
        "lifetime": read_csv(root / "lifetime_curves.csv"),
        "transition": read_csv(root / "transition_12m.csv"),
        "transition_summary": read_csv(root / "transition_summary.csv"),
        "backtesting_split": read_csv(root / "backtesting_split.csv"),
        "backtesting_year": read_csv(root / "backtesting_rating_year.csv"),
    }


def run_pd_from_controls(*, scorecard_run_id: str, max_lifetime_horizon: int) -> str:
    """Call the existing PD backend with the selected parent scorecard run."""
    from ifrs9.pd.framework import run_pd_framework

    run_id = f"pd_ui_{scorecard_run_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    result = run_pd_framework(
        repo_root=repo_root(),
        scorecard_run_id=scorecard_run_id,
        run_id=run_id,
        force=False,
    )
    _ = max_lifetime_horizon
    return result.run_id
