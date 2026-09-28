"""Configuration for LGD framework runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DiscountingConfig(BaseModel):
    """Discounting method configuration."""

    method: Literal["effective_rate_proxy"]
    fallback_rate: float | None = None


class LGDTargetConfig(BaseModel):
    """Model target bounding configuration."""

    lower_bound: float
    upper_bound: float
    cure_fallback_lgd: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> LGDTargetConfig:
        """Validate target bounds."""
        if self.lower_bound >= self.upper_bound:
            msg = "LGD lower_bound must be below upper_bound"
            raise ValueError(msg)
        return self


class DevelopmentConfig(BaseModel):
    """Temporal split cutoffs."""

    train_end: date
    validation_end: date
    oot_end: date


class SegmentationConfig(BaseModel):
    """LGD segmentation controls."""

    variables: list[str]
    minimum_resolved_defaults: int = Field(ge=1)


class CureModelConfig(BaseModel):
    """Cure model settings."""

    type: Literal["logistic_regression"]
    max_iter: int = Field(ge=100)


class SeverityModelConfig(BaseModel):
    """Non-cure severity model settings."""

    type: Literal["linear_regression"]
    bounded_link: Literal["linear_clipped_target"]


class ModelConfig(BaseModel):
    """LGD model settings."""

    cure: CureModelConfig
    severity: SeverityModelConfig


class DownturnConfig(BaseModel):
    """Downturn LGD overlay settings."""

    method: Literal["none", "historical_stress", "quantile_overlay"]
    stress_start: date
    stress_end: date
    quantile: float = Field(ge=0, le=1)


class FeatureConfig(BaseModel):
    """Predictor feature lists."""

    numeric: list[str]
    categorical: list[str]


class RunConfig(BaseModel):
    """Artifact path settings."""

    artifact_root: str
    model_root: str


class LGDFrameworkConfig(BaseModel):
    """Configurable LGD framework settings."""

    version: str
    default_definition: Literal["existing"]
    parent_pd_run: str
    resolution_horizon_months: int = Field(ge=1)
    cure_probation_months: int = Field(ge=1)
    discounting: DiscountingConfig
    target: LGDTargetConfig
    development: DevelopmentConfig
    segmentation: SegmentationConfig
    models: ModelConfig
    downturn: DownturnConfig
    features: FeatureConfig
    run: RunConfig


class LGDConfigFile(BaseModel):
    """Top-level YAML shape."""

    lgd_framework: LGDFrameworkConfig


def load_lgd_config(repo_root: Path, path: Path | None = None) -> LGDFrameworkConfig:
    """Load LGD framework configuration."""
    config_path = path or repo_root / "config" / "lgd.yaml"
    with config_path.open() as stream:
        parsed = LGDConfigFile.model_validate(yaml.safe_load(stream))
    return parsed.lgd_framework
