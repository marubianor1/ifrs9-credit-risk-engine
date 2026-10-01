"""Configuration for SICR and staging framework runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class OtherCreditDeteriorationConfig(BaseModel):
    """Non-PD SICR deterioration controls."""

    enabled: bool
    include_modification: bool
    include_assistance: bool


class SICRTriggerConfig(BaseModel):
    """SICR trigger thresholds."""

    relative_pd_increase: float | None = Field(default=None, gt=0)
    absolute_pd_increase: float | None = Field(default=None, ge=0)
    rating_downgrade_notches: int | None = Field(default=None, ge=1)
    dpd_backstop_months: int | None = Field(default=None, ge=1)
    other_credit_deterioration: OtherCreditDeteriorationConfig


class Stage3Config(BaseModel):
    """Stage 3 controls."""

    default_flag: bool


class LowCreditRiskConfig(BaseModel):
    """Low-credit-risk exemption controls."""

    enabled: bool
    max_rating: str | None = None


class CureConfig(BaseModel):
    """Stage cure controls."""

    probation_months: int = Field(ge=0)
    stage2_to_1_rule: Literal["immediate", "consecutive_months_without_sicr"]


class SensitivityConfig(BaseModel):
    """Deterministic threshold sensitivity grid."""

    relative_pd_increase: list[float]
    rating_downgrade_notches: list[int]
    dpd_backstop_months: list[int]


class StagingOutputConfig(BaseModel):
    """Output locations."""

    staging_path: str
    artifact_root: str


class StagingFrameworkConfig(BaseModel):
    """Configurable SICR and staging settings."""

    version: str
    parent_pd_run: str
    sicr: SICRTriggerConfig
    stage3: Stage3Config
    low_credit_risk: LowCreditRiskConfig
    cure: CureConfig
    sensitivity: SensitivityConfig
    output: StagingOutputConfig


class StagingConfigFile(BaseModel):
    """Top-level YAML shape."""

    staging_framework: StagingFrameworkConfig


def load_staging_config(repo_root: Path, path: Path | None = None) -> StagingFrameworkConfig:
    """Load SICR and staging configuration."""
    config_path = path or repo_root / "config" / "sicr.yaml"
    with config_path.open() as stream:
        parsed = StagingConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.staging_framework
