"""Configuration for EAD framework runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class EADMethodConfig(BaseModel):
    """Candidate and selected EAD method controls."""

    selected: Literal["current_balance", "contractual_amortization", "empirical_model"]
    candidates: list[Literal["current_balance", "contractual_amortization", "empirical_model"]]

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> EADMethodConfig:
        """Ensure the selected method is part of the comparison set."""
        if self.selected not in self.candidates:
            msg = "Selected EAD method must be included in candidates"
            raise ValueError(msg)
        return self


class EADDevelopmentConfig(BaseModel):
    """Temporal split cutoffs."""

    train_end: date
    validation_end: date
    oot_end: date


class EADHorizonConfig(BaseModel):
    """Horizon profile controls."""

    max_months: int = Field(ge=0, le=600)
    profile_population: Literal["latest_per_loan", "all_observations"]
    max_profile_observations: int | None = Field(default=None, ge=1)


class EADTargetPopulationConfig(BaseModel):
    """Observation population controls for EAD target construction."""

    include_non_default_exits: bool
    max_non_default_exit_observations: int | None = Field(default=None, ge=1)


class EADAmortizationConfig(BaseModel):
    """Contractual amortization assumptions."""

    interest_rate_floor: float = Field(ge=0)
    stop_at_maturity: bool


class EADEmpiricalModelConfig(BaseModel):
    """Empirical EAD model controls."""

    type: Literal["linear_regression"]
    target: Literal["ead_ratio"]
    lower_bound: float = Field(ge=0)
    upper_bound: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> EADEmpiricalModelConfig:
        """Validate empirical target bounds."""
        if self.lower_bound >= self.upper_bound:
            msg = "EAD empirical lower_bound must be below upper_bound"
            raise ValueError(msg)
        return self


class EADFeatureConfig(BaseModel):
    """Predictor feature lists."""

    numeric: list[str]
    categorical: list[str]


class EADReportingConfig(BaseModel):
    """Reporting metric controls."""

    mape_denominator_floor: float = Field(gt=0)


class EADRunConfig(BaseModel):
    """Artifact path settings."""

    artifact_root: str
    model_root: str


class EADFrameworkConfig(BaseModel):
    """Configurable EAD framework settings."""

    version: str
    default_definition: Literal["existing"]
    parent_pd_run: str
    methods: EADMethodConfig
    development: EADDevelopmentConfig
    horizons: EADHorizonConfig
    target_population: EADTargetPopulationConfig
    amortization: EADAmortizationConfig
    empirical_model: EADEmpiricalModelConfig
    features: EADFeatureConfig
    reporting: EADReportingConfig
    run: EADRunConfig


class EADConfigFile(BaseModel):
    """Top-level YAML shape."""

    ead_framework: EADFrameworkConfig


def load_ead_config(repo_root: Path, path: Path | None = None) -> EADFrameworkConfig:
    """Load EAD framework configuration."""
    config_path = path or repo_root / "config" / "ead.yaml"
    with config_path.open() as stream:
        parsed = EADConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.ead_framework
