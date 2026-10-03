from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from app.services.ead import BASELINE_METHOD, load_ead_artifacts
from app.services.ecl import ecl_kpis, load_ecl_artifacts
from app.services.lgd import BASELINE_RUN, available_lgd_runs, lgd_governance_status
from app.services.monitoring import load_monitoring_thresholds, oe_status, status_band
from app.services.pd import available_pd_runs, load_pd_artifacts
from app.services.scenario_lab import (
    cached_result_exists,
    load_lab_config,
    run_or_load_scenario,
    scenario_config_from_controls,
    scenario_hash,
    scenario_run_id,
)
from app.services.scorecard import available_scorecard_runs, load_scorecard_artifacts
from app.services.sicr import run_staging_simulation, staging_overrides_from_controls


def test_ecl_artifact_loading_and_kpis() -> None:
    artifacts = load_ecl_artifacts("ecl_v1")
    kpis = ecl_kpis(artifacts["stage"], artifacts["scenario"])

    assert kpis["total_ead"] > 0
    assert kpis["weighted_ecl"] > 0
    assert 0 < kpis["coverage_ratio"] < 1
    assert not artifacts["downturn"].empty


def test_run_discovery_and_artifact_loading() -> None:
    scorecard_runs = available_scorecard_runs()
    pd_runs = available_pd_runs()

    assert "behavioural_qe_v1" in scorecard_runs
    assert "pd_behavioural_qe_v1" in pd_runs
    assert not load_scorecard_artifacts("behavioural_qe_v1")["metrics"].empty
    assert not load_pd_artifacts("pd_behavioural_qe_v1")["rating_summary"].empty


def test_lgd_run_discovery_and_governance_status() -> None:
    status = lgd_governance_status()

    assert BASELINE_RUN in available_lgd_runs()
    assert status["production_baseline"] == BASELINE_RUN
    assert "lgd_v1_3" in status["rejected_challengers"]


def test_ead_artifact_loading() -> None:
    artifacts = load_ead_artifacts("ead_v1")

    assert BASELINE_METHOD == "contractual_amortization"
    assert not artifacts["backtest_split"].empty
    assert artifacts["ratio_distribution"]["observable_defaults"].iloc[0] > 0


def test_staging_config_conversion_and_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.sicr as service

    staged = pd.DataFrame(
        {
            "loan_id": ["A", "B", "C"],
            "as_of_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
            "stage": [1, 2, 3],
            "ead_current": [100.0, 200.0, 300.0],
            "origination_baseline_available": [True, True, True],
            "pd_relative_change": [1.0, 2.5, 1.0],
            "pd_absolute_change": [0.0, 0.02, 0.0],
            "rating_notch_change": [0, 3, 0],
            "sicr_dpd_backstop": [False, True, True],
            "sicr_other_credit_deterioration": [False, False, False],
        }
    )
    monkeypatch.setattr(service, "load_staging_dataset", lambda: staged)
    overrides = staging_overrides_from_controls(
        relative_pd=2.0,
        absolute_pd=None,
        rating_downgrade=3,
        dpd_backstop=1,
        cure_probation=3,
    )
    result = run_staging_simulation(overrides)

    assert set(result) == {"sensitivity", "stage_distribution"}
    assert not result["stage_distribution"].empty


def test_monitoring_threshold_statuses() -> None:
    thresholds = load_monitoring_thresholds()

    assert status_band(0.80, thresholds["scorecard_auc"], higher_is_better=True) == "GREEN"
    assert status_band(0.30, thresholds["scorecard_psi"], higher_is_better=False) == "RED"
    assert oe_status(1.0, thresholds["oe_ratio"]) == "GREEN"
    assert oe_status(2.0, thresholds["oe_ratio"]) == "RED"


def test_scenario_config_hash_and_validation() -> None:
    config = scenario_config_from_controls(
        name="unit",
        description="Unit scenario",
        macro={
            "unemployment_shock": 1.0,
            "hpi_shock": -5.0,
            "gdp_shock": -1.0,
            "mortgage_rate_shock": 0.5,
            "rho_override": None,
            "scenario_weights": {"UPSIDE": 0.2, "BASE": 0.5, "DOWNSIDE": 0.3},
        },
        staging={
            "relative_pd_threshold": 2.0,
            "absolute_pd_threshold": None,
            "rating_downgrade_threshold": 3,
            "dpd_backstop": 1,
            "stage2_cure_probation": 3,
        },
        portfolio={
            "rating_downgrade_share": 0.01,
            "delinquency_1m_share": 0.01,
            "delinquency_2m_share": 0.0,
            "delinquency_3m_default_share": 0.0,
            "seed": 7,
        },
        lgd={"method": "structural_base", "manual_overlay_pp": 0.0},
        ead={"multiplier": 1.0},
    )

    assert scenario_hash(config) == scenario_hash(config)
    assert scenario_run_id(config).startswith("scenario_ui_unit_")
    with pytest.raises(ValueError):
        scenario_config_from_controls(
            name="bad",
            description="Bad weights",
            macro={"scenario_weights": {"UPSIDE": 0.2, "BASE": 0.2, "DOWNSIDE": 0.2}},
            staging={},
            portfolio={},
            lgd={},
            ead={},
        )


def test_cached_scenario_lookup_uses_existing_artifacts() -> None:
    config = load_lab_config().presets["baseline"]

    assert cached_result_exists(config) is False


def test_run_or_load_scenario_uses_cache_and_mocks_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.scenario_lab as service

    config = load_lab_config().presets["baseline"].model_copy(update={"name": "unit_cache"})
    calls = {"count": 0}

    def fake_artifact_dir(run_id: str) -> Path:
        return tmp_path / run_id

    def fake_run_scenario(**kwargs):
        calls["count"] += 1
        root = fake_artifact_dir(kwargs["run_id"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps({"run_id": kwargs["run_id"]}) + "\n")
        (root / "config.json").write_text(json.dumps(config.model_dump(), default=str) + "\n")
        pd.DataFrame([{"stage": 1}]).to_csv(root / "stage_comparison.csv", index=False)
        pd.DataFrame([{"rating": "R1"}]).to_csv(root / "rating_comparison.csv", index=False)
        pd.DataFrame([{"driver": "PD / macro effect", "effect": 0.0}]).to_csv(
            root / "driver_waterfall.csv",
            index=False,
        )

    monkeypatch.setattr(service, "scenario_artifact_dir", fake_artifact_dir)
    monkeypatch.setattr(service, "run_scenario", fake_run_scenario)

    _, _, cached_first = run_or_load_scenario(config)
    _, _, cached_second = run_or_load_scenario(config)

    assert cached_first is False
    assert cached_second is True
    assert calls["count"] == 1
