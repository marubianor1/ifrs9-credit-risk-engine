"""EAD artifact services for the Streamlit app."""

from __future__ import annotations

import pandas as pd

from app.services.artifacts import (
    artifact_path,
    discover_runs,
    read_csv,
    read_json,
    read_parquet,
    repo_root,
)
from app.services.runtime import is_cloud_demo

BASELINE_METHOD = "contractual_amortization"


def available_ead_runs() -> list[str]:
    """Return available EAD run IDs."""
    return discover_runs("ead")


def load_ead_artifacts(run_id: str = "ead_v1") -> dict[str, pd.DataFrame | dict]:
    """Load EAD framework artifacts."""
    root = artifact_path("artifacts", "ead", run_id)
    return {
        "run": read_json(root / "run.json"),
        "population": read_csv(root / "population_summary.csv"),
        "ratio_distribution": read_csv(root / "ead_ratio_distribution.csv"),
        "backtest_split": read_csv(root / "backtest_by_split.csv"),
        "backtest_segment": read_csv(root / "backtest_by_segment.csv"),
        "profile_validation": read_csv(root / "profile_validation.csv"),
        "target_reconciliation": read_csv(root / "target_reconciliation.csv"),
        "profiles": read_csv(root / "ead_profiles.csv")
        if is_cloud_demo()
        else read_parquet(root / "ead_profiles.parquet"),
    }


def run_ead_from_controls() -> str:
    """Call the existing EAD backend explicitly."""
    from ifrs9.ead.framework import run_ead_framework

    result = run_ead_framework(repo_root=repo_root(), run_id=None, force=False)
    return result.run_id
