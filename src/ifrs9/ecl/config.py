"""Configuration for IFRS 9 ECL engine runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ECLOutputConfig(BaseModel):
    """ECL output paths."""

    gold_path: str
    artifact_root: str


class ECLConfig(BaseModel):
    """Configurable ECL engine settings."""

    version: str
    reporting_date: date
    pd_run: str
    lgd_run: str
    ead_run: str
    staging_run: str
    ead_method: Literal["contractual_amortization"]
    lgd_method: Literal["structural_base", "downturn_sensitivity"]
    discount_rate: Literal["effective_rate_proxy"]
    scenario_weighting: bool
    max_lifetime_months: int = Field(ge=12, le=600)
    output: ECLOutputConfig


class ECLConfigFile(BaseModel):
    """Top-level YAML shape."""

    ecl_engine: ECLConfig


def load_ecl_config(repo_root: Path, path: Path | None = None) -> ECLConfig:
    """Load ECL engine configuration."""
    config_path = path or repo_root / "config" / "ecl.yaml"
    with config_path.open() as stream:
        parsed = ECLConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.ecl_engine
