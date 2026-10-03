"""ECL artifact services for the Streamlit app."""

from __future__ import annotations

import pandas as pd

from app.services.artifacts import artifact_path, read_csv, read_json, repo_root
from ifrs9.ecl.engine import ECLRunResult, run_ecl


def load_ecl_artifacts(run_id: str = "ecl_v1") -> dict[str, pd.DataFrame | dict]:
    """Load compact ECL outputs for dashboard pages."""
    root = artifact_path("artifacts", "ecl", run_id)
    return {
        "run": read_json(root / "run.json"),
        "stage": read_csv(root / "ecl_by_stage.csv"),
        "rating": read_csv(root / "ecl_by_rating.csv"),
        "scenario": read_csv(root / "ecl_by_scenario.csv"),
        "vintage": read_csv(root / "ecl_by_vintage.csv"),
        "warnings": read_csv(root / "alignment_warnings.csv"),
        "reconciliation": read_csv(root / "scenario_reconciliation.csv"),
        "downturn": read_csv(root / "downturn_lgd_sensitivity.csv"),
    }


def ecl_kpis(stage: pd.DataFrame, scenario: pd.DataFrame) -> dict[str, float]:
    """Summarize ECL KPI values from persisted artifacts."""
    total_ead = float(stage["total_ead"].sum())
    weighted_ecl = float(scenario.loc[scenario["scenario"].eq("WEIGHTED"), "ecl"].iloc[0])
    return {
        "total_ead": total_ead,
        "weighted_ecl": weighted_ecl,
        "coverage_ratio": weighted_ecl / total_ead if total_ead else 0.0,
    }


def run_ecl_from_ui() -> ECLRunResult:
    """Run the ECL backend from an explicit UI action."""
    return run_ecl(repo_root=repo_root())
