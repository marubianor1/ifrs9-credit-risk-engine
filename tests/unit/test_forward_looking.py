from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9.macro.config import load_forward_looking_config
from ifrs9.macro.forward_looking import (
    _scenario_lifetime_curves,
    _weighted_pd_curves,
    calibrate_rho,
    infer_systematic_factor,
    load_forward_looking_run,
    run_forward_looking,
    simulate_macro_scenario,
    vasicek_pit_pd,
)


def _config():
    return load_forward_looking_config(Path.cwd())


def test_vasicek_formula_inverts_and_uses_better_factor_as_lower_pd() -> None:
    rho = 0.12
    ttc = 0.02
    observed = 0.05
    factor = infer_systematic_factor(observed, ttc, rho)
    reconstructed = vasicek_pit_pd(ttc, factor, rho)

    assert reconstructed == pytest.approx(observed)
    assert vasicek_pit_pd(ttc, 1.0, rho) < vasicek_pit_pd(ttc, -1.0, rho)
    with pytest.raises(ValueError):
        vasicek_pit_pd(ttc, 0.0, 1.0)


def test_rho_calibration_respects_override_and_bounds() -> None:
    config = _config()
    rates = np.array([0.004, 0.006, 0.012, 0.02, 0.008])
    rho, grid = calibrate_rho(rates, 0.008, config)

    assert config.vasicek.rho_min <= rho <= config.vasicek.rho_max
    assert grid["selected"].sum() == 1

    override = config.model_copy(
        deep=True,
        update={"vasicek": config.vasicek.model_copy(update={"rho_override": 0.09})},
    )
    rho_override, _ = calibrate_rho(rates, 0.008, override)
    assert rho_override == pytest.approx(0.09)


def test_scenario_weighted_pd_calculation() -> None:
    scenario_pd = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-31", "2026-03-31"]),
            "rating": ["R1", "R1"],
            "ttc_pd": [0.01, 0.01],
            "scenario": ["BASE", "DOWNSIDE"],
            "scenario_weight": [0.7, 0.3],
            "pit_pd_12m": [0.02, 0.06],
            "weighted_pd_12m": [0.014, 0.018],
        }
    )
    weighted = _weighted_pd_curves(scenario_pd)

    assert weighted["weighted_pd_12m"].iloc[0] == pytest.approx(0.032)
    assert weighted["scenario_weight_sum"].iloc[0] == pytest.approx(1.0)


def test_invalid_scenario_weights_are_rejected() -> None:
    config = _config()
    payload = config.model_dump()
    payload["scenarios"]["weights"]["BASE"] = 0.20

    with pytest.raises(ValueError, match="Scenario weights must sum"):
        type(config).model_validate(payload)


def test_lifetime_adjustment_preserves_monotonic_curves() -> None:
    config = _config()
    base = pd.DataFrame(
        {
            "rating": ["R1"] * 24,
            "month": list(range(1, 25)),
            "conditional_pd": [0.001] * 24,
            "hazard": [0.001] * 24,
            "marginal_pd": [0.001] * 24,
            "cumulative_pd": np.linspace(0.001, 0.024, 24),
            "survival_probability": 1 - np.linspace(0.001, 0.024, 24),
        }
    )
    scenario_pd = pd.DataFrame(
        {
            "scenario": ["BASE"],
            "date": pd.to_datetime(["2026-03-31"]),
            "rating": ["R1"],
            "pit_pd_12m": [0.04],
            "scenario_weight": [1.0],
        }
    )

    adjusted = _scenario_lifetime_curves(base, scenario_pd, config)

    assert adjusted["cumulative_pd"].between(0, 1).all()
    assert adjusted["cumulative_pd"].is_monotonic_increasing
    assert adjusted["survival_probability"].is_monotonic_decreasing
    assert adjusted.loc[adjusted["month"] == 12, "cumulative_pd"].iloc[0] == pytest.approx(0.04)


def _write_forward_repo(repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config").mkdir()
    config_text = (Path.cwd() / "config" / "forward_looking.yaml").read_text()
    (repo / "config" / "forward_looking.yaml").write_text(config_text)
    pd_base = repo / "artifacts" / "pd" / "small_pd"
    pd_base.mkdir(parents=True)
    macro_dir = repo / "data" / "raw" / "macro"
    macro_dir.mkdir(parents=True)
    dates = pd.date_range("2018-03-31", periods=16, freq="QE")
    macro = pd.DataFrame(
        {
            "date": dates,
            "unemployment_rate": np.linspace(4.0, 8.0, len(dates)),
            "unemployment_rate_change": [0.0] + [0.25] * (len(dates) - 1),
            "house_price_index_yoy": np.linspace(5.0, -6.0, len(dates)),
            "real_gdp_yoy": np.linspace(3.0, -2.0, len(dates)),
            "mortgage_rate": np.linspace(3.5, 6.0, len(dates)),
            "mortgage_rate_change": [0.0] + [0.15] * (len(dates) - 1),
        }
    )
    macro.to_csv(macro_dir / "fred_quarterly.csv", index=False)
    rows = []
    for quarter_index, date in enumerate(pd.date_range("2018-03-31", periods=16, freq="QE")):
        bad_count = min(quarter_index // 4 + 1, 5)
        for index in range(10):
            rows.append(
                {
                    "loan_id": f"L{quarter_index:02d}{index:02d}",
                    "as_of_date": date,
                    "rating": "R1" if index < 5 else "R2",
                    "target": int(index < bad_count),
                }
            )
    predictions = pd.DataFrame(rows)
    predictions.to_parquet(pd_base / "pd_predictions.parquet", index=False)
    pd.DataFrame({"portfolio_ttc_pd": [0.02]}).to_csv(pd_base / "portfolio_ttc.csv", index=False)
    pd.DataFrame(
        {"rating": ["R1", "R2"], "rating_ttc_pd": [0.01, 0.04], "observations": [80, 80]}
    ).to_csv(pd_base / "rating_ttc.csv", index=False)
    lifetime = pd.DataFrame(
        {
            "rating": ["R1"] * 24 + ["R2"] * 24,
            "month": list(range(1, 25)) * 2,
            "conditional_pd": [0.001] * 24 + [0.003] * 24,
            "hazard": [0.001] * 24 + [0.003] * 24,
            "marginal_pd": [0.001] * 48,
            "cumulative_pd": list(np.linspace(0.001, 0.024, 24))
            + list(np.linspace(0.003, 0.072, 24)),
            "survival_probability": list(1 - np.linspace(0.001, 0.024, 24))
            + list(1 - np.linspace(0.003, 0.072, 24)),
        }
    )
    lifetime.to_csv(pd_base / "lifetime_curves.csv", index=False)


def test_forward_run_persists_and_scenarios_are_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_forward_repo(repo)
    config = load_forward_looking_config(repo)

    result = run_forward_looking(
        repo_root=repo,
        pd_run_id="small_pd",
        run_id="forward_small",
        force=True,
    )
    first = simulate_macro_scenario(
        config,
        config.scenarios.shocks,
        repo_root=repo,
        parent_pd_run="small_pd",
    )
    second = simulate_macro_scenario(
        config,
        config.scenarios.shocks,
        repo_root=repo,
        parent_pd_run="small_pd",
    )
    downside = first.scenario_pd_curves[first.scenario_pd_curves["scenario"] == "DOWNSIDE"]
    upside = first.scenario_pd_curves[first.scenario_pd_curves["scenario"] == "UPSIDE"]

    assert Path(result.artifact_path, "scenario_pd_curves.csv").exists()
    assert Path(result.artifact_path, "weighted_lifetime_curves.csv").exists()
    assert Path(result.model_path, "forward_looking_models.pkl").exists()
    assert load_forward_looking_run(repo, "forward_small")["parent_pd_run"] == "small_pd"
    pd.testing.assert_frame_equal(first.weighted_pd_curves, second.weighted_pd_curves)
    assert downside["pit_pd_12m"].mean() > upside["pit_pd_12m"].mean()
    assert first.weighted_pd_curves["scenario_weight_sum"].eq(1.0).all()

    with pytest.raises(FileExistsError):
        run_forward_looking(repo_root=repo, pd_run_id="small_pd", run_id="forward_small")
