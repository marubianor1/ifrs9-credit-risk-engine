from __future__ import annotations

from pathlib import Path

import pandas as pd

from ifrs9.lgd.forward_looking import (
    LGDForwardLookingConfig,
    LGDForwardOutputConfig,
    LGDForwardSensitivityConfig,
    _apply_historical_overlay,
    _fit_overlay,
    _historical_lgd_regime_dataset,
    _scenario_lgd_by_rating,
    _scenario_sensitivity,
    _weighted_scenario_lgd,
    load_lgd_forward_run,
)


def _config() -> LGDForwardLookingConfig:
    return LGDForwardLookingConfig(
        version="v1",
        method="macro_overlay",
        parent_lgd_run="parent",
        scenario_run="scenario",
        calibration_split="VALIDATION",
        overlay_floor=0.0,
        overlay_cap=None,
        scenario_horizon_quarters=2,
        macro_features=[
            "unemployment_rate",
            "house_price_index_yoy",
            "real_gdp_yoy",
            "mortgage_rate",
            "systematic_factor",
        ],
        stress_weights={
            "unemployment_rate": 1.0,
            "house_price_index_yoy": -1.0,
            "real_gdp_yoy": -0.5,
            "mortgage_rate": 0.25,
            "systematic_factor": -0.5,
        },
        sensitivity=LGDForwardSensitivityConfig(
            hpi_shocks=[-1.0, 0.0, 1.0],
            unemployment_shocks=[-1.0, 0.0, 1.0],
        ),
        output=LGDForwardOutputConfig(
            artifact_root="artifacts/lgd_forward_looking",
            model_root="models/lgd_forward_looking",
        ),
    )


def _episodes() -> pd.DataFrame:
    rows = []
    quarters = [
        ("2018-03-01", "TRAIN", 0.30, 0.20),
        ("2018-06-01", "TRAIN", 0.35, 0.25),
        ("2018-09-01", "TRAIN", 0.60, 0.30),
        ("2019-03-01", "VALIDATION", 0.40, 0.30),
        ("2019-06-01", "VALIDATION", 0.50, 0.40),
        ("2020-03-01", "OOT", 0.45, 0.35),
    ]
    for index, (date, split, actual, predicted) in enumerate(quarters):
        rows.append(
            {
                "loan_id": f"L{index}",
                "default_date": pd.Timestamp(date),
                "split": split,
                "rating_at_default": "R1" if index % 2 == 0 else "R2",
                "ead_at_default": 100.0,
                "resolved_flag": True,
                "cured_flag": index % 2 == 0,
                "realized_lgd_model_target": actual,
                "predicted_lgd": predicted,
                "predicted_cure_probability": 0.6,
                "predicted_cure_lgd": 0.10,
                "predicted_non_cure_lgd": 0.50,
                "estimated_loan_to_value_at_default": 80 + index,
                "current_upb_to_original_upb_at_default": 0.8,
                "default_due_to_delinquency": True,
                "default_due_to_credit_event": False,
                "default_year": pd.Timestamp(date).year,
            }
        )
    return pd.DataFrame(rows)


def _macro() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-03-31",
                    "2018-06-30",
                    "2018-09-30",
                    "2019-03-31",
                    "2019-06-30",
                    "2020-03-31",
                ]
            ),
            "unemployment_rate": [4.0, 4.5, 6.0, 4.2, 4.8, 5.0],
            "house_price_index_yoy": [5.0, 4.0, 1.0, 4.5, 3.0, 2.0],
            "real_gdp_yoy": [3.0, 2.5, 0.0, 2.8, 1.8, 1.0],
            "mortgage_rate": [4.0, 4.2, 5.0, 4.1, 4.5, 4.8],
            "systematic_factor": [1.0, 0.5, -1.0, 0.8, 0.1, -0.5],
        }
    )


def test_overlay_fit_excludes_oot_and_uses_validation_intercept() -> None:
    config = _config()
    historical = _historical_lgd_regime_dataset(_episodes(), _macro(), config)
    overlay = _fit_overlay(historical, config)
    oot = historical[historical["split"] == "OOT"].copy()
    changed = historical.copy()
    changed.loc[changed["split"] == "OOT", "structural_oe_ratio"] = 99.0

    assert overlay == _fit_overlay(changed, config)
    assert overlay["slope"] >= 0
    assert not oot.empty


def test_branch_reconciliation_and_deterministic_overlay() -> None:
    config = _config()
    historical = _historical_lgd_regime_dataset(_episodes(), _macro(), config)
    overlay = _fit_overlay(historical, config)

    first = _apply_historical_overlay(_episodes(), _macro(), overlay, config)
    second = _apply_historical_overlay(_episodes(), _macro(), overlay, config)

    assert first["lgd_forward_adjusted"].equals(second["lgd_forward_adjusted"])
    assert first["lgd_forward_adjusted"].equals(first["branch_reconciled_lgd"])
    assert first["lgd_forward_adjusted"].between(0, 1).all()


def test_economic_sign_convention_and_scenario_ordering() -> None:
    config = _config()
    historical = _historical_lgd_regime_dataset(_episodes(), _macro(), config)
    overlay = _fit_overlay(historical, config)
    rating_base = pd.DataFrame(
        {
            "rating": ["R1"],
            "lgd_base_structural": [0.3],
            "predicted_cure_probability": [0.5],
            "predicted_cure_lgd": [0.1],
            "predicted_non_cure_lgd": [0.5],
            "observed_lgd": [0.3],
            "rows": [10],
        }
    )
    scenario_macro = pd.DataFrame(
        {
            "scenario": ["UPSIDE", "BASE", "DOWNSIDE"],
            "date": pd.to_datetime(["2025-03-31"] * 3),
            "scenario_weight": [0.2, 0.5, 0.3],
            "unemployment_rate": [3.0, 4.0, 6.0],
            "house_price_index_yoy": [6.0, 3.0, -1.0],
            "real_gdp_yoy": [3.0, 2.0, -1.0],
            "mortgage_rate": [4.0, 5.0, 6.0],
            "systematic_factor": [1.0, 0.0, -1.0],
        }
    )

    scenario = _scenario_lgd_by_rating(rating_base, scenario_macro, overlay, config)
    ordered = scenario.set_index("scenario")["lgd_scenario"]

    assert ordered["DOWNSIDE"] >= ordered["BASE"] >= ordered["UPSIDE"]
    assert _weighted_scenario_lgd(scenario)["scenario_weight_sum"].iloc[0] == 1.0


def test_sensitivity_and_run_reload(tmp_path: Path) -> None:
    config = _config()
    historical = _historical_lgd_regime_dataset(_episodes(), _macro(), config)
    overlay = _fit_overlay(historical, config)
    rating_base = pd.DataFrame(
        {
            "rating": ["R1"],
            "lgd_base_structural": [0.3],
            "predicted_cure_probability": [0.5],
            "predicted_cure_lgd": [0.1],
            "predicted_non_cure_lgd": [0.5],
            "observed_lgd": [0.3],
            "rows": [10],
        }
    )
    scenario_macro = pd.DataFrame(
        {
            "scenario": ["BASE"],
            "date": pd.to_datetime(["2025-03-31"]),
            "scenario_weight": [1.0],
            "unemployment_rate": [4.0],
            "house_price_index_yoy": [3.0],
            "real_gdp_yoy": [2.0],
            "mortgage_rate": [5.0],
            "systematic_factor": [0.0],
        }
    )

    sensitivity = _scenario_sensitivity(rating_base, scenario_macro, overlay, config)
    assert len(sensitivity) == 9

    repo = tmp_path / "repo"
    run_dir = repo / "artifacts" / "lgd_forward_looking" / "lgd_fl_v1"
    run_dir.mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "config" / "lgd_forward_looking.yaml").write_text(
        Path.cwd().joinpath("config/lgd_forward_looking.yaml").read_text()
    )
    (run_dir / "run.json").write_text('{"run_id": "lgd_fl_v1"}\n')
    assert load_lgd_forward_run(repo, "lgd_fl_v1")["run_id"] == "lgd_fl_v1"
