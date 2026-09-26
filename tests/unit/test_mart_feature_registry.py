from __future__ import annotations

from pathlib import Path

from ifrs9.mart.feature_registry import load_default_registry


def test_feature_registry_excludes_restricted_and_technical_features_by_default() -> None:
    registry = load_default_registry(Path.cwd())

    pd_features = registry.get_features_for("pd")

    assert "actual_loss" not in pd_features
    assert "_gold_processed_at" not in pd_features
    assert "original_credit_score" in pd_features
    assert "max_delinquency_months_to_date" in pd_features


def test_feature_registry_can_include_restricted_features_when_explicitly_requested() -> None:
    registry = load_default_registry(Path.cwd())

    restricted = registry.get_features_for("lgd_outcome", include_restricted=True)
    default = registry.get_features_for("lgd_outcome")

    assert "actual_loss" in restricted
    assert "actual_loss" not in default
    assert registry.counts_by_leakage()["OUTCOME_RESTRICTED"] > 0
    assert registry.counts_by_availability()["CURRENT_PERIOD"] > 0
