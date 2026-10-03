"""Configuration for the IFRS 9 scenario lab backend."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ParentRunConfig(BaseModel):
    """Accepted parent run identifiers."""

    pd: str
    lgd: str
    ead: str
    staging: str
    ecl: str


class MacroShockConfig(BaseModel):
    """Scenario controls that affect forward-looking PD curves."""

    unemployment_shock: float = 0.0
    hpi_shock: float = 0.0
    gdp_shock: float = 0.0
    mortgage_rate_shock: float = 0.0
    rho_override: float | None = Field(default=None, gt=0, lt=1)
    scenario_weights: dict[str, float] | None = None


class StagingShockConfig(BaseModel):
    """Scenario controls that affect SICR/staging allocation."""

    relative_pd_threshold: float | None = Field(default=None, gt=0)
    absolute_pd_threshold: float | None = Field(default=None, ge=0)
    rating_downgrade_threshold: int | None = Field(default=None, ge=1)
    dpd_backstop: int | None = Field(default=None, ge=1)
    stage2_cure_probation: int | None = Field(default=None, ge=0)


class PortfolioShockConfig(BaseModel):
    """Deterministic portfolio-state overlays."""

    rating_downgrade_share: float = Field(default=0.0, ge=0, le=1)
    delinquency_1m_share: float = Field(default=0.0, ge=0, le=1)
    delinquency_2m_share: float = Field(default=0.0, ge=0, le=1)
    delinquency_3m_default_share: float = Field(default=0.0, ge=0, le=1)
    seed: int = 1729


class LGDShockConfig(BaseModel):
    """Scenario controls that affect LGD."""

    method: Literal["structural_base", "downturn_sensitivity"] = "structural_base"
    manual_overlay_pp: float = 0.0


class EADShockConfig(BaseModel):
    """Scenario controls that affect EAD."""

    multiplier: float = Field(default=1.0, ge=0)


class ScenarioConfig(BaseModel):
    """Single scenario preset or run configuration."""

    name: str
    description: str = ""
    macro: MacroShockConfig = Field(default_factory=MacroShockConfig)
    staging: StagingShockConfig = Field(default_factory=StagingShockConfig)
    portfolio: PortfolioShockConfig = Field(default_factory=PortfolioShockConfig)
    lgd: LGDShockConfig = Field(default_factory=LGDShockConfig)
    ead: EADShockConfig = Field(default_factory=EADShockConfig)

    @model_validator(mode="after")
    def validate_weights(self) -> ScenarioConfig:
        """Validate optional scenario weights."""
        weights = self.macro.scenario_weights
        if weights is not None and abs(sum(weights.values()) - 1.0) > 1e-8:
            msg = f"Scenario weights must sum to 1.0, got {sum(weights.values()):.8f}"
            raise ValueError(msg)
        return self


class ScenarioLabOutputConfig(BaseModel):
    """Scenario lab output location."""

    artifact_root: str


class ScenarioLabConfig(BaseModel):
    """Top-level scenario lab configuration."""

    version: str
    reporting_date: date
    parents: ParentRunConfig
    default_preset: str
    output: ScenarioLabOutputConfig
    presets: dict[str, ScenarioConfig]


class ScenarioLabConfigFile(BaseModel):
    """YAML file shape."""

    scenario_lab: ScenarioLabConfig


def load_scenario_lab_config(repo_root: Path, path: Path | None = None) -> ScenarioLabConfig:
    """Load scenario lab configuration from YAML."""
    config_path = path or repo_root / "config" / "scenario_lab.yaml"
    with config_path.open() as stream:
        parsed = ScenarioLabConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.scenario_lab
