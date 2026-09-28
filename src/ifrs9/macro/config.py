"""Configuration for forward-looking PD scenario runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class MacroSeriesConfig(BaseModel):
    """Single public macro series configuration."""

    fred_id: str
    frequency: str
    aggregation: Literal["mean", "last"]
    transform: Literal["level", "yoy_pct"]


class MacroConfig(BaseModel):
    """Public macro data source configuration."""

    source: Literal["fred"]
    raw_cache_dir: str
    transformed_path: str
    start_date: date
    series: dict[str, MacroSeriesConfig]
    feature_columns: list[str]


class VasicekConfig(BaseModel):
    """One-factor Vasicek settings."""

    rho_override: float | None = Field(default=None, gt=0, lt=1)
    rho_min: float = Field(gt=0, lt=1)
    rho_max: float = Field(gt=0, lt=1)
    rho_grid_size: int = Field(ge=10)
    clipping_epsilon: float = Field(gt=0, lt=0.01)

    @model_validator(mode="after")
    def validate_bounds(self) -> VasicekConfig:
        """Validate rho search bounds."""
        if self.rho_min >= self.rho_max:
            msg = "rho_min must be lower than rho_max"
            raise ValueError(msg)
        return self


class SatelliteConfig(BaseModel):
    """Macro satellite challenger settings."""

    regularization_strength: float = Field(gt=0)
    target: Literal["observed_default_rate"]


class ScenarioConfig(BaseModel):
    """Forward-looking scenario settings."""

    mode: Literal["historical_shock", "custom"]
    horizon_quarters: int = Field(ge=1)
    start_date: date
    base_growth_scale: float = Field(ge=0)
    weights: dict[str, float]
    shocks: dict[str, dict[str, float]]

    @model_validator(mode="after")
    def validate_weights(self) -> ScenarioConfig:
        """Validate scenario probabilities and names."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-8:
            msg = f"Scenario weights must sum to 1.0, got {total:.8f}"
            raise ValueError(msg)
        if set(self.weights) != set(self.shocks):
            msg = "Scenario weights and shocks must contain the same scenario names"
            raise ValueError(msg)
        if any(weight < 0 for weight in self.weights.values()):
            msg = "Scenario weights must be non-negative"
            raise ValueError(msg)
        return self


class LifetimeForwardConfig(BaseModel):
    """Lifetime PD forward-looking adjustment settings."""

    adjustment_method: Literal["conditional_hazard_ratio"]
    max_horizon_months: int = Field(ge=12)


class BacktestingConfig(BaseModel):
    """Historical backtesting window settings."""

    stress_start: date
    stress_end: date


class ForwardRunConfig(BaseModel):
    """Artifact path configuration."""

    artifact_root: str
    model_root: str


class ForwardLookingConfig(BaseModel):
    """Configurable forward-looking PD framework settings."""

    version: str
    parent_pd_run: str
    method: Literal["vasicek", "macro_satellite"]
    challenger_method: Literal["vasicek", "macro_satellite"]
    macro: MacroConfig
    vasicek: VasicekConfig
    satellite: SatelliteConfig
    scenarios: ScenarioConfig
    lifetime: LifetimeForwardConfig
    backtesting: BacktestingConfig
    run: ForwardRunConfig


class ForwardLookingConfigFile(BaseModel):
    """Top-level YAML shape."""

    forward_looking: ForwardLookingConfig


def load_forward_looking_config(
    repo_root: Path,
    path: Path | None = None,
) -> ForwardLookingConfig:
    """Load forward-looking PD configuration."""
    config_path = path or repo_root / "config" / "forward_looking.yaml"
    with config_path.open() as stream:
        parsed = ForwardLookingConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.forward_looking
