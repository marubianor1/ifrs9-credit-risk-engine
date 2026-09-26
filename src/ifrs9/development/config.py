"""Configuration objects for temporal development samples."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Population = Literal["application", "behavioural"]
SnapshotFrequency = Literal["monthly", "quarter_end", "year_end"]
SamplingStrategy = Literal["none", "random_nondefault", "stratified_nondefault"]


class DateWindow(BaseModel):
    """Inclusive date window used for temporal splits."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> DateWindow:
        """Validate that the start date is not after the end date."""
        if self.start > self.end:
            msg = f"Invalid date window: {self.start} is after {self.end}"
            raise ValueError(msg)
        return self


class SplitWindows(BaseModel):
    """Temporal split windows."""

    train: DateWindow
    validation: DateWindow
    oot: DateWindow

    @model_validator(mode="after")
    def validate_non_overlapping(self) -> SplitWindows:
        """Validate that configured date windows do not overlap."""
        windows = [
            ("train", self.train),
            ("validation", self.validation),
            ("oot", self.oot),
        ]
        for index, (left_name, left) in enumerate(windows):
            for right_name, right in windows[index + 1 :]:
                if left.start <= right.end and right.start <= left.end:
                    msg = f"Split windows overlap: {left_name} and {right_name}"
                    raise ValueError(msg)
        return self


class SamplingConfig(BaseModel):
    """Non-default sampling configuration."""

    strategy: SamplingStrategy
    seed: int
    nondefault_sample_fraction: float = Field(gt=0, le=1)
    minimum_nondefaults_per_stratum: int = Field(ge=0)


class DevelopmentConfig(BaseModel):
    """Configuration for development sample construction."""

    version: str
    reporting_cutoff: date
    binary_12m_full_horizon_cutoff: date
    model_use: str = "pd"
    population: Population
    snapshot_frequency: SnapshotFrequency
    split_windows: SplitWindows
    sampling: SamplingConfig
    loan_disjoint: bool
    notes: list[str]


class DevelopmentConfigFile(BaseModel):
    """Top-level YAML file shape."""

    development_sample: DevelopmentConfig


def load_development_config(repo_root: Path, path: Path | None = None) -> DevelopmentConfig:
    """Load development sample configuration."""
    config_path = path or repo_root / "config" / "development_samples.yaml"
    with config_path.open() as stream:
        parsed = DevelopmentConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.development_sample
