"""Configuration for calibrated PD framework runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CalibrationConfig(BaseModel):
    """Calibration model configuration."""

    method: Literal["none", "logistic_recalibration", "isotonic"]
    fit_split: str
    raw_pd_column: str


class RatingScaleConfig(BaseModel):
    """Rating master scale configuration."""

    grade_count: int = Field(ge=2)
    labels: list[str]
    minimum_population_pct: float = Field(gt=0, lt=1)
    minimum_defaults: int = Field(ge=0)


class TTCConfig(BaseModel):
    """Through-the-cycle anchor configuration."""

    anchor_start: date
    anchor_end: date
    weighting: Literal["observation_weighted"]


class LifetimeConfig(BaseModel):
    """Lifetime term-structure configuration."""

    max_horizon_months: int = Field(ge=12)
    smoothing: float = Field(ge=0)
    reconcile_first_12m: bool


class TransitionConfig(BaseModel):
    """Rating transition configuration."""

    frequencies_months: list[int]
    default_state: str


class BacktestingConfig(BaseModel):
    """Backtesting configuration."""

    insufficient_default_threshold: int = Field(ge=0)


class PDRunConfig(BaseModel):
    """Artifact path configuration."""

    artifact_root: str
    model_root: str


class PDFrameworkConfig(BaseModel):
    """Configurable PD framework settings."""

    version: str
    primary_scorecard_run: str
    secondary_scorecard_runs: list[str]
    calibration: CalibrationConfig
    rating_master_scale: RatingScaleConfig
    ttc: TTCConfig
    lifetime: LifetimeConfig
    transitions: TransitionConfig
    backtesting: BacktestingConfig
    run: PDRunConfig


class PDConfigFile(BaseModel):
    """Top-level YAML shape."""

    pd_framework: PDFrameworkConfig


def load_pd_config(repo_root: Path, path: Path | None = None) -> PDFrameworkConfig:
    """Load PD framework configuration."""
    config_path = path or repo_root / "config" / "pd.yaml"
    with config_path.open() as stream:
        parsed = PDConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.pd_framework
