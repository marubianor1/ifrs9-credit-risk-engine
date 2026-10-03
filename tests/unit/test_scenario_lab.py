from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ifrs9.ecl.config import ECLConfig, ECLOutputConfig
from ifrs9.scenario_lab.config import (
    EADShockConfig,
    MacroShockConfig,
    ParentRunConfig,
    PortfolioShockConfig,
    ScenarioConfig,
    ScenarioLabConfig,
    ScenarioLabOutputConfig,
    StagingShockConfig,
)
from ifrs9.scenario_lab.engine import (
    _apply_ead_controls,
    _apply_macro_controls,
    _apply_portfolio_shock,
    _apply_staging_controls,
    _driver_waterfall,
    _macro_pd_scale,
    _score_state,
)


def _ecl_config() -> ECLConfig:
    return ECLConfig(
        version="v1",
        reporting_date=pd.Timestamp("2025-03-01").date(),
        pd_run="forward_looking_pd_behavioural_qe_v1",
        lgd_run="lgd_v1_2",
        ead_run="ead_v1",
        staging_run="sicr_v1_1",
        ead_method="contractual_amortization",
        lgd_method="structural_base",
        discount_rate="effective_rate_proxy",
        scenario_weighting=True,
        max_lifetime_months=360,
        output=ECLOutputConfig(gold_path="data/gold/freddie/ecl", artifact_root="artifacts/ecl"),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["A", "B", "C", "D"],
            "as_of_date": [pd.Timestamp("2025-03-01")] * 4,
            "default_episode_id": [None, None, None, "D-1"],
            "stage": [1, 1, 2, 3],
            "base_stage": [1, 1, 2, 3],
            "rating": ["R1", "R2", "R3", "R5"],
            "rating_current": ["R1", "R2", "R3", "R5"],
            "rating_origination": ["R1", "R1", "R2", "R5"],
            "rating_notch_change": [0, 1, 1, 0],
            "ead": [100.0, 100.0, 100.0, 100.0],
            "lgd": [0.4, 0.4, 0.4, 0.4],
            "current_interest_rate": [0.0, 0.0, 0.0, 0.0],
            "remaining_months_to_maturity": [24, 24, 24, 24],
            "active_default_state": [False, False, False, True],
            "origination_baseline_available": [True, True, True, True],
            "pd_relative_change": [1.0, 1.4, 1.6, 1.0],
            "pd_absolute_change": [0.0, 0.01, 0.02, 0.0],
            "delinquency_months": [0, 0, 1, 3],
            "sicr_relative_pd": [False, False, True, False],
            "sicr_absolute_pd": [False, False, False, False],
            "sicr_rating_downgrade": [False, False, False, False],
            "sicr_dpd_backstop": [False, False, True, True],
            "sicr_other_credit_deterioration": [False, False, False, False],
        }
    )


def _curves() -> dict[tuple[str, str], pd.DataFrame]:
    rows = []
    for scenario, weight, pd_scale in [
        ("BASE", 0.6, 1.0),
        ("UPSIDE", 0.2, 0.8),
        ("DOWNSIDE", 0.2, 1.4),
    ]:
        for rating in ["R1", "R2", "R3", "R4", "R5"]:
            for month in range(1, 25):
                rows.append(
                    {
                        "scenario": scenario,
                        "rating": rating,
                        "month": month,
                        "marginal_pd": 0.01 * pd_scale,
                        "scenario_weight": weight,
                    }
                )
    frame = pd.DataFrame(rows)
    return {
        key: group.copy()
        for key, group in frame.groupby(["scenario", "rating"], observed=True)
    }


def test_baseline_artifact_reproduces_ecl_v1_summary() -> None:
    root = Path.cwd()
    loan_level = pd.read_parquet(root / "data" / "gold" / "freddie" / "ecl" / "part-ecl.parquet")
    scenario = pd.read_csv(root / "artifacts" / "ecl" / "ecl_v1" / "ecl_by_scenario.csv")

    assert np.isclose(
        loan_level["ecl_weighted"].sum(),
        scenario.loc[scenario["scenario"].eq("WEIGHTED"), "ecl"].iloc[0],
    )


def test_changing_weights_does_not_mutate_parent_curves() -> None:
    curves = _curves()
    stressed = _apply_macro_controls(
        curves,
        MacroShockConfig(scenario_weights={"UPSIDE": 0.1, "BASE": 0.4, "DOWNSIDE": 0.5}),
        Path.cwd(),
        _lab_stub(),
    )

    assert curves[("BASE", "R1")]["scenario_weight"].iloc[0] == 0.6
    assert stressed[("BASE", "R1")]["scenario_weight"].iloc[0] == 0.4


def test_worse_macro_increases_pd_and_ecl() -> None:
    curves = _curves()
    stressed = _apply_macro_controls(
        curves,
        MacroShockConfig(unemployment_shock=2.0, hpi_shock=-10.0, gdp_shock=-2.0),
        Path.cwd(),
        _lab_stub(),
    )
    base = _score_state(_frame(), curves, _ecl_config())
    stress = _score_state(_frame(), stressed, _ecl_config())

    assert _macro_pd_scale(MacroShockConfig(unemployment_shock=2.0), Path.cwd(), _lab_stub()) > 1
    assert stress["ecl_weighted"].sum() > base["ecl_weighted"].sum()


def test_stricter_sicr_threshold_changes_staging() -> None:
    loose = _apply_staging_controls(
        _frame(),
        StagingShockConfig(relative_pd_threshold=1.5, dpd_backstop=1),
    )
    strict = _apply_staging_controls(
        _frame(),
        StagingShockConfig(relative_pd_threshold=3.0, dpd_backstop=2),
    )

    assert loose["stage"].eq(2).sum() > strict["stage"].eq(2).sum()


def test_ead_multiplier_behaves_correctly() -> None:
    stressed = _apply_ead_controls(_frame(), EADShockConfig(multiplier=1.25))

    assert np.isclose(stressed["ead"].sum(), _frame()["ead"].sum() * 1.25)


def test_deterministic_portfolio_shocks_and_canonical_frame_unchanged() -> None:
    source = _frame()
    before = source.copy(deep=True)
    shock = PortfolioShockConfig(
        rating_downgrade_share=0.5,
        delinquency_1m_share=0.5,
        seed=7,
    )

    first = _apply_portfolio_shock(source, shock)
    second = _apply_portfolio_shock(source, shock)

    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_series_equal(first["rating"], second["rating"])
    pd.testing.assert_series_equal(first["delinquency_months"], second["delinquency_months"])


def test_waterfall_reconciles_exactly_on_parent_artifacts() -> None:
    root = Path.cwd()
    from ifrs9.ecl.engine import _pd_curves
    from ifrs9.scenario_lab.engine import _scenario_state

    config = _ecl_config()
    state = _scenario_state(root, config).head(250).copy()
    curves, _ = _pd_curves(root, config)
    baseline = _score_state(state, curves, config)
    scenario = ScenarioConfig(
        name="unit",
        macro=MacroShockConfig(unemployment_shock=1.0),
        staging=StagingShockConfig(relative_pd_threshold=1.5, dpd_backstop=1),
        portfolio=PortfolioShockConfig(rating_downgrade_share=0.05, seed=11),
        ead=EADShockConfig(multiplier=1.1),
    )
    waterfall = _driver_waterfall(root, config, state, baseline, scenario, _lab_stub())
    stressed_delta = waterfall["ending_ecl"].iloc[-1] - baseline["ecl_weighted"].sum()

    assert np.isclose(waterfall["effect"].sum(), stressed_delta)


def test_downturn_lgd_artifact_increases_ecl() -> None:
    sensitivity = pd.read_csv(
        Path.cwd() / "artifacts" / "ecl" / "ecl_v1" / "downturn_lgd_sensitivity.csv"
    )
    base = sensitivity.loc[sensitivity["lgd_method"].eq("structural_base"), "weighted_ecl"].iloc[0]
    downturn = sensitivity.loc[
        sensitivity["lgd_method"].eq("downturn_sensitivity"),
        "weighted_ecl",
    ].iloc[0]

    assert downturn > base


def _lab_stub():
    return ScenarioLabConfig(
        version="v1",
        reporting_date="2025-03-01",
        parents=ParentRunConfig(
            pd="forward_looking_pd_behavioural_qe_v1",
            lgd="lgd_v1_2",
            ead="ead_v1",
            staging="sicr_v1_1",
            ecl="ecl_v1",
        ),
        default_preset="baseline",
        output=ScenarioLabOutputConfig(artifact_root="artifacts/scenarios"),
        presets={},
    )
