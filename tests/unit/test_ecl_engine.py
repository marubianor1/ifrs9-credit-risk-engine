from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ifrs9.ead.framework import contractual_balance
from ifrs9.ecl.config import ECLConfig, ECLOutputConfig
from ifrs9.ecl.engine import _scenario_ecl, _scenario_weights, ecl_calendar_month


def _config() -> ECLConfig:
    return ECLConfig(
        version="v1",
        reporting_date=date(2025, 3, 1),
        pd_run="pd",
        lgd_run="lgd",
        ead_run="ead",
        staging_run="sicr",
        ead_method="contractual_amortization",
        lgd_method="structural_base",
        discount_rate="effective_rate_proxy",
        scenario_weighting=True,
        max_lifetime_months=360,
        output=ECLOutputConfig(gold_path="data/gold/freddie/ecl", artifact_root="artifacts/ecl"),
    )


def _curves() -> dict[tuple[str, str], pd.DataFrame]:
    rows = []
    scenarios = [("BASE", 0.6, 1.0), ("UPSIDE", 0.2, 0.5), ("DOWNSIDE", 0.2, 2.0)]
    for scenario, weight, scale in scenarios:
        for month in range(1, 25):
            rows.append(
                {
                    "scenario": scenario,
                    "rating": "R1",
                    "month": month,
                    "marginal_pd": 0.01 * scale,
                    "scenario_weight": weight,
                }
            )
    frame = pd.DataFrame(rows)
    return {
        (scenario, rating): group
        for (scenario, rating), group in frame.groupby(["scenario", "rating"], observed=True)
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["S1", "S2", "S3"],
            "stage": [1, 2, 3],
            "rating": ["R1", "R1", "R1"],
            "ead": [100.0, 100.0, 100.0],
            "lgd": [0.5, 0.5, 0.5],
            "current_interest_rate": [0.0, 0.0, 12.0],
            "remaining_months_to_maturity": [24, 24, 24],
        }
    )


def test_stage1_uses_max_12_months_and_stage2_uses_lifetime() -> None:
    ecl = _scenario_ecl(_frame(), _curves(), "BASE", _config())
    stage1_ead = contractual_balance(100.0, 0.0, 24.0, np.arange(1, 13))
    stage2_ead = contractual_balance(100.0, 0.0, 24.0, np.arange(1, 25))

    assert np.isclose(ecl.iloc[0], (stage1_ead * 0.01 * 0.5).sum())
    assert np.isclose(ecl.iloc[1], (stage2_ead * 0.01 * 0.5).sum())
    assert ecl.iloc[1] > ecl.iloc[0]


def test_reporting_date_is_horizon_zero_and_month_one_is_prospective() -> None:
    assert ecl_calendar_month(date(2025, 3, 1), 0) == pd.Timestamp("2025-03-01")
    assert ecl_calendar_month(date(2025, 3, 1), 1) == pd.Timestamp("2025-04-01")
    assert ecl_calendar_month(date(2025, 3, 1), 3) == pd.Timestamp("2025-06-01")


def test_quarterly_scenario_anchor_does_not_shift_month_one_pd() -> None:
    frame = _frame().iloc[[0]].copy()
    curves = {
        ("BASE", "R1"): pd.DataFrame(
            {
                "scenario": ["BASE", "BASE", "BASE"],
                "rating": ["R1", "R1", "R1"],
                "date": [pd.Timestamp("2025-06-30")] * 3,
                "month": [1, 2, 3],
                "marginal_pd": [0.07, 0.0, 0.0],
                "scenario_weight": [1.0, 1.0, 1.0],
            }
        )
    }
    ecl = _scenario_ecl(frame, curves, "BASE", _config())

    expected_month_one = contractual_balance(100.0, 0.0, 24.0, 1) * 0.07 * 0.5
    assert np.isclose(ecl.iloc[0], expected_month_one)


def test_stage1_uses_exactly_12_future_months() -> None:
    frame = _frame().iloc[[0]].copy()
    curves = {
        ("BASE", "R1"): pd.DataFrame(
            {
                "scenario": ["BASE"] * 13,
                "rating": ["R1"] * 13,
                "month": list(range(1, 14)),
                "marginal_pd": [0.0] * 12 + [0.99],
                "scenario_weight": [1.0] * 13,
            }
        )
    }

    ecl = _scenario_ecl(frame, curves, "BASE", _config())

    assert np.isclose(ecl.iloc[0], 0.0)


def test_stage2_starts_at_month_one_and_runs_to_maturity() -> None:
    frame = _frame().iloc[[1]].copy()
    frame["remaining_months_to_maturity"] = 2
    curves = {
        ("BASE", "R1"): pd.DataFrame(
            {
                "scenario": ["BASE", "BASE", "BASE"],
                "rating": ["R1", "R1", "R1"],
                "month": [1, 2, 3],
                "marginal_pd": [0.03, 0.04, 0.99],
                "scenario_weight": [1.0, 1.0, 1.0],
            }
        )
    }

    ecl = _scenario_ecl(frame, curves, "BASE", _config())
    expected = (
        contractual_balance(100.0, 0.0, 2.0, 1) * 0.03 * 0.5
        + contractual_balance(100.0, 0.0, 2.0, 2) * 0.04 * 0.5
    )

    assert np.isclose(ecl.iloc[0], expected)


def test_stage3_uses_pd_100_percent_without_ordinary_pd() -> None:
    ecl = _scenario_ecl(_frame(), _curves(), "BASE", _config())

    assert np.isclose(ecl.iloc[2], 50.0)


def test_marginal_pd_scenario_weighting_and_no_negative_ecl() -> None:
    curves = _curves()
    weights = _scenario_weights(curves)
    frame = _frame()
    base = _scenario_ecl(frame, curves, "BASE", _config())
    upside = _scenario_ecl(frame, curves, "UPSIDE", _config())
    downside = _scenario_ecl(frame, curves, "DOWNSIDE", _config())
    weighted = upside * weights["UPSIDE"] + base * weights["BASE"] + downside * weights["DOWNSIDE"]

    assert np.isclose(sum(weights.values()), 1.0)
    assert (weighted >= 0).all()
    assert weighted.iloc[0] > upside.iloc[0]
    assert weighted.iloc[0] < downside.iloc[0]


def test_discounting_reduces_stage2_ecl() -> None:
    frame = _frame()
    undiscounted = _scenario_ecl(
        frame.assign(current_interest_rate=0.0),
        _curves(),
        "BASE",
        _config(),
    )
    discounted = _scenario_ecl(
        frame.assign(current_interest_rate=12.0),
        _curves(),
        "BASE",
        _config(),
    )

    assert discounted.iloc[1] < undiscounted.iloc[1]


def test_downturn_sensitivity_config_is_separate_from_baseline() -> None:
    baseline = _config()
    sensitivity = baseline.model_copy(update={"lgd_method": "downturn_sensitivity"})

    assert baseline.lgd_method == "structural_base"
    assert sensitivity.lgd_method == "downturn_sensitivity"
