"""Scenario Lab services for Streamlit controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.artifacts import artifact_path, read_csv, read_json, repo_root
from ifrs9.scenario_lab import run_scenario
from ifrs9.scenario_lab.config import (
    EADShockConfig,
    LGDShockConfig,
    MacroShockConfig,
    PortfolioShockConfig,
    ScenarioConfig,
    StagingShockConfig,
    load_scenario_lab_config,
)


def load_lab_config():
    """Load the Scenario Lab YAML config."""
    return load_scenario_lab_config(repo_root())


def scenario_config_from_controls(
    *,
    name: str,
    description: str,
    macro: dict[str, Any],
    staging: dict[str, Any],
    portfolio: dict[str, Any],
    lgd: dict[str, Any],
    ead: dict[str, Any],
) -> ScenarioConfig:
    """Convert UI controls into a validated ScenarioConfig."""
    return ScenarioConfig(
        name=name,
        description=description,
        macro=MacroShockConfig(**macro),
        staging=StagingShockConfig(**staging),
        portfolio=PortfolioShockConfig(**portfolio),
        lgd=LGDShockConfig(**lgd),
        ead=EADShockConfig(**ead),
    )


def scenario_hash(config: ScenarioConfig) -> str:
    """Return a stable hash for a ScenarioConfig."""
    payload = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def scenario_run_id(config: ScenarioConfig) -> str:
    """Build a deterministic scenario run ID."""
    return f"scenario_ui_{config.name}_{scenario_hash(config)}"


def scenario_artifact_dir(run_id: str) -> Path:
    """Return the artifact directory for a scenario run."""
    return artifact_path("artifacts", "scenarios", run_id)


def load_scenario_result(run_id: str) -> dict[str, pd.DataFrame | dict]:
    """Load a persisted Scenario Lab run."""
    root = scenario_artifact_dir(run_id)
    return {
        "summary": read_json(root / "summary.json"),
        "stage": read_csv(root / "stage_comparison.csv"),
        "rating": read_csv(root / "rating_comparison.csv"),
        "waterfall": read_csv(root / "driver_waterfall.csv"),
        "config": read_json(root / "config.json"),
    }


def cached_result_exists(config: ScenarioConfig) -> bool:
    """Return whether an identical scenario config already has artifacts."""
    return (scenario_artifact_dir(scenario_run_id(config)) / "summary.json").exists()


def run_or_load_scenario(
    config: ScenarioConfig,
) -> tuple[str, dict[str, pd.DataFrame | dict], bool]:
    """Reuse an existing scenario artifact or run the Scenario Lab backend."""
    run_id = scenario_run_id(config)
    cached = cached_result_exists(config)
    if not cached:
        run_scenario(preset="baseline", overrides=config.model_dump(), run_id=run_id, force=True)
    return run_id, load_scenario_result(run_id), cached
