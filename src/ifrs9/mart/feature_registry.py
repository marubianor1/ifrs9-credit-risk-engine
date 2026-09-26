"""Feature registry and leakage-aware selectors for the point-in-time mart."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

AllowedUse = Literal[
    "application_scoring",
    "behavioural_scoring",
    "pd",
    "sicr",
    "ead",
    "lgd_predictor",
    "lgd_outcome",
    "reporting",
]
LeakageRisk = Literal["SAFE_ORIGINATION", "SAFE_AS_OF_DATE", "OUTCOME_RESTRICTED", "TECHNICAL_ONLY"]

SAFE_FOR_MODELLING = {"SAFE_ORIGINATION", "SAFE_AS_OF_DATE"}


class FeatureDefinition(BaseModel):
    """Metadata for one point-in-time analytical field."""

    name: str
    description: str
    source: str
    category: str
    availability: str
    transformation: str
    window: str
    allowed_uses: list[AllowedUse]
    leakage_risk: LeakageRisk


class FeatureRegistry(BaseModel):
    """Point-in-time feature registry."""

    version: str
    leakage_classes: list[LeakageRisk]
    features: list[FeatureDefinition]

    @classmethod
    def load(cls, path: Path) -> FeatureRegistry:
        """Load a feature registry from YAML."""
        with path.open() as stream:
            return cls.model_validate(yaml.safe_load(stream))

    def get_features_for(self, use: AllowedUse, *, include_restricted: bool = False) -> list[str]:
        """Return feature names allowed for a use, excluding leakage-prone fields by default."""
        return [
            feature.name
            for feature in self.features
            if use in feature.allowed_uses
            and (include_restricted or feature.leakage_risk in SAFE_FOR_MODELLING)
        ]

    def get_by_availability(self, availability: str) -> list[str]:
        """Return feature names by availability class."""
        return [feature.name for feature in self.features if feature.availability == availability]

    def counts_by_leakage(self) -> dict[str, int]:
        """Return counts by leakage class."""
        counts: dict[str, int] = {}
        for feature in self.features:
            counts[feature.leakage_risk] = counts.get(feature.leakage_risk, 0) + 1
        return counts

    def counts_by_availability(self) -> dict[str, int]:
        """Return counts by availability class."""
        counts: dict[str, int] = {}
        for feature in self.features:
            counts[feature.availability] = counts.get(feature.availability, 0) + 1
        return counts


def load_default_registry(repo_root: Path) -> FeatureRegistry:
    """Load the repository's point-in-time feature registry."""
    return FeatureRegistry.load(repo_root / "config" / "features" / "point_in_time_features.yaml")

