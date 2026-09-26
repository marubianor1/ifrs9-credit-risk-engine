"""Configuration objects for logistic scorecard experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Population = Literal["application", "behavioural"]


class DateWindow(BaseModel):
    """Inclusive date window for a temporal split."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> DateWindow:
        """Validate window ordering."""
        if self.start > self.end:
            msg = f"Invalid date window: {self.start} is after {self.end}"
            raise ValueError(msg)
        return self


class BinningConfig(BaseModel):
    """WOE binning configuration."""

    max_bins: int = Field(ge=2)
    min_bin_pct: float = Field(gt=0, lt=1)
    enforce_monotonic: bool
    rare_category_pct: float = Field(gt=0, lt=1)
    smoothing: float = Field(gt=0)
    suspicious_iv_threshold: float = Field(gt=0)


class ScoreScalingConfig(BaseModel):
    """Traditional score scaling configuration."""

    base_score: float
    pdo: float = Field(gt=0)
    base_odds_good_to_bad: float = Field(gt=0)
    band_count: int = Field(ge=2)


class MetricsConfig(BaseModel):
    """Metric configuration."""

    confusion_cutoff: float = Field(gt=0, lt=1)
    deciles: int = Field(ge=2)


class RunConfig(BaseModel):
    """Run storage configuration."""

    artifact_root: str
    model_root: str


class ScorecardConfig(BaseModel):
    """Logistic scorecard experiment configuration."""

    model_type: str
    population: Population
    snapshot_frequency: Literal["monthly", "quarter_end", "year_end"]
    sampling_strategy: Literal["none", "random_nondefault", "stratified_nondefault"]
    target: str
    train: DateWindow
    validation: DateWindow
    oot: DateWindow
    binning: BinningConfig
    score_scaling: ScoreScalingConfig
    metrics: MetricsConfig
    run: RunConfig


class ScorecardConfigFile(BaseModel):
    """Top-level scorecard YAML shape."""

    scorecard: ScorecardConfig


def load_scorecard_config(repo_root: Path, path: Path | None = None) -> ScorecardConfig:
    """Load scorecard configuration."""
    config_path = path or repo_root / "config" / "models" / "scorecard.yaml"
    with config_path.open() as stream:
        parsed = ScorecardConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.scorecard
