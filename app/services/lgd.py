"""LGD artifact services for the Streamlit app."""

from __future__ import annotations

import pandas as pd

from app.services.artifacts import artifact_path, discover_runs, read_csv, read_json, read_parquet
from app.services.runtime import is_cloud_demo

BASELINE_RUN = "lgd_v1_2"
REJECTED_CHALLENGERS = ["lgd_v1_3", "lgd_fl_v1", "lgd_fl_v2"]


def available_lgd_runs() -> list[str]:
    """Return available LGD run IDs."""
    runs = discover_runs("lgd")
    preferred = [run for run in [BASELINE_RUN, "lgd_v1_3"] if run in runs]
    return preferred + [run for run in runs if run not in preferred]


def lgd_governance_status() -> dict[str, object]:
    """Return project LGD governance labels."""
    return {
        "production_baseline": BASELINE_RUN,
        "rejected_challengers": REJECTED_CHALLENGERS,
        "reason": (
            "lgd_v1_2 is retained as the project baseline because challenger runs showed "
            "weaker temporal/OOT stability or insufficient macro relationship support."
        ),
    }


def load_lgd_artifacts(run_id: str = BASELINE_RUN) -> dict[str, pd.DataFrame | dict]:
    """Load LGD framework artifacts for one run."""
    root = artifact_path("artifacts", "lgd", run_id)
    payload = {
        "run": read_json(root / "run.json"),
        "backtesting": read_csv(root / "backtesting_split.csv"),
        "combined": read_csv(root / "combined_backtest.csv"),
        "components": read_csv(root / "component_decomposition.csv"),
        "by_rating": read_csv(root / "observed_predicted_by_rating.csv"),
        "by_year": read_csv(root / "observed_predicted_by_default_year.csv"),
        "downturn": read_csv(root / "downturn_overlay.csv"),
        "recovery_timing": read_csv(root / "recovery_timing.csv"),
        "resolution": read_csv(root / "resolution_distribution.csv"),
        "model_metrics": read_csv(root / "model_metrics.csv"),
    }
    payload["episodes"] = (
        read_csv(root / "population_summary.csv")
        if is_cloud_demo()
        else read_parquet(root / "lgd_episodes.parquet")
    )
    return payload


def load_lgd_forward_artifacts(run_id: str) -> dict[str, pd.DataFrame | dict]:
    """Load forward-looking LGD challenger artifacts."""
    root = artifact_path("artifacts", "lgd_forward_looking", run_id)
    return {
        "run": read_json(root / "run.json"),
        "scenario_by_rating": read_csv(root / "scenario_lgd_by_rating.csv"),
        "weighted": read_csv(root / "probability_weighted_lgd.csv"),
        "sensitivity": read_csv(root / "scenario_sensitivity.csv"),
        "diagnostics": read_csv(root / "macro_lgd_relationships.csv"),
    }


def lgd_population_summary(episodes: pd.DataFrame) -> dict[str, float]:
    """Summarize LGD episode population."""
    if {"episodes", "resolved", "cure_rate"}.issubset(episodes.columns):
        row = episodes.iloc[0]
        return {
            "episodes": float(row["episodes"]),
            "resolved": float(row["resolved"]),
            "cure_rate": float(row["cure_rate"]),
        }
    resolved = episodes["resolved"].fillna(False) if "resolved" in episodes else pd.Series(False)
    cure = episodes["resolution_type"].astype(str).str.upper().eq("CURE")
    return {
        "episodes": float(len(episodes)),
        "resolved": float(resolved.sum()) if len(resolved) else 0.0,
        "cure_rate": float(cure.mean()) if len(episodes) else 0.0,
    }
