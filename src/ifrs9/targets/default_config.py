"""Configuration loading for default definitions and PD targets."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CureConfig(BaseModel):
    """Cure rule configuration."""

    probation_months: int = Field(ge=1)
    require_consecutive_observed_months: bool
    reporting_gaps_break_probation: bool


class DefaultDefinitionConfig(BaseModel):
    """Default definition and target-construction configuration."""

    version: str
    source: str
    reporting_cutoff: date
    delinquency_threshold_months: int = Field(ge=1)
    target_horizon_months: int = Field(ge=1)
    cure: CureConfig
    event_hierarchy: list[str]
    delinquency_status_credit_events: dict[str, str]
    zero_balance_default_codes: dict[str, str]
    zero_balance_non_default_exit_codes: dict[str, str]
    excluded_from_default: list[str]
    notes: list[str]

    @property
    def delinquency_status_codes_sql(self) -> str:
        """Return SQL literal list for default delinquency status codes."""
        return _sql_string_list(self.delinquency_status_credit_events)

    @property
    def default_zero_balance_codes_sql(self) -> str:
        """Return SQL literal list for default zero-balance codes."""
        return _sql_string_list(self.zero_balance_default_codes)

    @property
    def non_default_zero_balance_codes_sql(self) -> str:
        """Return SQL literal list for non-default zero-balance exit codes."""
        return _sql_string_list(self.zero_balance_non_default_exit_codes)


class DefaultDefinitionFile(BaseModel):
    """Top-level YAML file shape."""

    default_definition: DefaultDefinitionConfig


def _sql_string_list(values: dict[str, str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def load_default_definition(repo_root: Path) -> DefaultDefinitionConfig:
    """Load the repository default definition YAML."""
    with (repo_root / "config" / "default_definition.yaml").open() as stream:
        parsed = DefaultDefinitionFile.model_validate(yaml.safe_load(stream))
    return parsed.default_definition
