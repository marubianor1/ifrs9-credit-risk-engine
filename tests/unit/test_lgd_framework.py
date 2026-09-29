from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9.lgd.config import load_lgd_config
from ifrs9.lgd.framework import (
    OUTCOME_EXCLUDED_COLUMNS,
    _actual_loss_reconciliation,
    _add_lgd_targets,
    _combined_model_selection,
    _component_audit,
    _component_decomposition,
    _fit_probability_recalibrator,
    _temporal_training_weights,
    _validated_predictors,
    build_lgd_episode_dataset,
    load_lgd_run,
    run_lgd_framework,
)


def _copy_config(repo: Path) -> None:
    (repo / "config").mkdir()
    shutil.copy(Path.cwd() / "config" / "lgd.yaml", repo / "config" / "lgd.yaml")


def _minimal_month_row(
    loan_id: str,
    date: str,
    *,
    zero_balance_code: str | None = None,
    actual_loss: float | None = None,
    current_upb: float = 100_000.0,
) -> dict[str, object]:
    return {
        "loan_id": loan_id,
        "vintage_year": 2020,
        "as_of_date": pd.Timestamp(date).date(),
        "current_actual_upb": current_upb,
        "current_upb_lag_1": 100_000.0,
        "current_interest_rate": 6.0,
        "estimated_loan_to_value": 85,
        "current_upb_to_original_upb": 0.9,
        "delinquency_months": 3,
        "max_delinquency_months_to_date": 3,
        "months_delinquent_to_date": 3.0,
        "months_since_last_delinquency": 0,
        "months_since_origination": 36,
        "ever_modified_to_date": False,
        "ever_assistance_to_date": False,
        "zero_balance_code": zero_balance_code,
        "net_sale_proceeds": "70000" if zero_balance_code else None,
        "mi_recoveries": 5_000.0 if zero_balance_code else None,
        "non_mi_recoveries": 2_000.0 if zero_balance_code else None,
        "total_expenses": 4_000.0 if zero_balance_code else None,
        "delinquent_accrued_interest": 1_000.0 if zero_balance_code else None,
        "actual_loss": actual_loss,
        "cumulative_modification_costs": 500.0 if zero_balance_code else None,
        "current_period_modification_costs": 100.0 if zero_balance_code else None,
        "bankruptcy_cramdown_costs": 0.0,
        "legal_costs": 800.0 if zero_balance_code else None,
        "maintenance_and_preservation_costs": 600.0 if zero_balance_code else None,
        "taxes_and_insurance": 700.0 if zero_balance_code else None,
        "miscellaneous_expenses": 200.0 if zero_balance_code else None,
    }


def _write_repo(repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    _copy_config(repo)
    base = repo / "data" / "gold" / "freddie"
    (base / "targets" / "default_events").mkdir(parents=True)
    (base / "loan_month" / "vintage_year=2020").mkdir(parents=True)
    (base / "loan_static" / "vintage_year=2020").mkdir(parents=True)
    silver = repo / "data" / "silver" / "freddie" / "performance" / "vintage_year=2020"
    silver.mkdir(parents=True)
    (repo / "artifacts" / "pd" / "pd_behavioural_qe_v1").mkdir(parents=True)

    events = []
    months = []
    static = []
    pd_rows = []
    for index in range(18):
        loan_id = f"L{index:03d}"
        year = 2018 if index < 10 else 2020 if index < 14 else 2023
        default_date = f"{year}-03-01"
        cured = index % 3 == 0
        unresolved = index in {8, 13, 17}
        terminal_code = None if unresolved or cured else ("09" if index % 2 else "03")
        cure_date = f"{year}-06-01" if cured else None
        events.append(
            {
                "loan_id": loan_id,
                "vintage_year": 2020,
                "default_episode_id": 1,
                "default_entry_date": pd.Timestamp(default_date).date(),
                "cure_date": pd.Timestamp(cure_date).date() if cure_date else None,
                "default_reason": "DELINQUENCY_3_PLUS",
                "default_due_to_delinquency": True,
                "default_due_to_credit_event": False,
                "redefault_flag": False,
                "redefault_date": None,
                "_target_processed_at": "test",
            }
        )
        months.append(_minimal_month_row(loan_id, default_date))
        if terminal_code:
            months.append(
                _minimal_month_row(
                    loan_id,
                    f"{year}-09-01",
                    zero_balance_code=terminal_code,
                    actual_loss=25_000.0 + index * 100,
                    current_upb=0.0,
                )
            )
        static.append(
            {
                "loan_id": loan_id,
                "vintage_year": 2020,
                "original_credit_score": 700 - index,
                "original_dti": 32,
                "original_ltv": 80,
                "original_cltv": 82,
                "original_upb": 110_000.0,
                "original_interest_rate": 5.5,
                "mortgage_insurance_percentage": 20,
                "property_state": "CA",
                "property_type": "SF",
                "occupancy_status": "P",
                "loan_purpose": "P",
                "channel": "R",
                "_gold_processed_at": "test",
            }
        )
        pd_rows.append(
            {
                "loan_id": loan_id,
                "as_of_date": pd.Timestamp(default_date).date(),
                "rating": "R1" if index < 9 else "R2",
            }
        )
    pd.DataFrame(events).to_parquet(
        base / "targets" / "default_events" / "part.parquet",
        index=False,
    )
    pd.DataFrame(months).to_parquet(
        base / "loan_month" / "vintage_year=2020" / "part.parquet",
        index=False,
    )
    pd.DataFrame(static).to_parquet(
        base / "loan_static" / "vintage_year=2020" / "part.parquet",
        index=False,
    )
    silver_rows = pd.DataFrame(months).rename(columns={"as_of_date": "period"})
    silver_rows["_source_year"] = silver_rows["vintage_year"]
    silver_rows["zero_balance_removal_upb"] = np.where(
        silver_rows["zero_balance_code"].notna(),
        100_000.0,
        np.nan,
    )
    silver_rows["delinquent_accrued_interest"] = np.where(
        silver_rows["zero_balance_code"].notna(),
        1_000.0,
        np.nan,
    )
    silver_rows.to_parquet(silver / "part.parquet", index=False)
    pd.DataFrame(pd_rows).to_parquet(
        repo / "artifacts" / "pd" / "pd_behavioural_qe_v1" / "pd_predictions.parquet",
        index=False,
    )


def test_episode_population_one_row_per_default_and_censoring(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo)
    config = load_lgd_config(repo)

    episodes = build_lgd_episode_dataset(repo, config)

    assert len(episodes) == 18
    assert episodes[["loan_id", "default_episode_id"]].drop_duplicates().shape[0] == 18
    assert episodes["cured_flag"].sum() == 6
    assert (~episodes["resolved_flag"]).sum() == 3
    assert episodes["rating_at_default"].notna().all()
    assert episodes.loc[episodes["actual_loss"].notna(), "zero_balance_removal_upb"].notna().all()


def test_discounted_cashflows_lgd_bounds_and_predictor_exclusions() -> None:
    config = load_lgd_config(Path.cwd())
    frame = pd.DataFrame(
        {
            "ead_at_default": [100.0, 100.0, 100.0],
            "current_interest_rate_at_default": [12.0, 12.0, 12.0],
            "original_interest_rate": [10.0, 10.0, 10.0],
            "months_to_resolution": [12, 12, 12],
            "resolved_flag": [True, True, True],
            "cured_flag": [False, False, False],
            "zero_balance_removal_upb": [100.0, 100.0, 100.0],
            "net_sale_proceeds": [-120.0, 0.0, -20.0],
            "mi_recoveries": [0.0, 0.0, 0.0],
            "non_mi_recoveries": [0.0, 0.0, 0.0],
            "total_expenses": [0.0, 150.0, 10.0],
            "delinquent_accrued_interest": [0.0, 0.0, 0.0],
            "cumulative_modification_costs": [0.0, 0.0, 0.0],
            "bankruptcy_cramdown_costs": [0.0, 0.0, 0.0],
        }
    )

    output = _add_lgd_targets(frame, config)

    assert output["realized_lgd_raw"].iloc[0] < 0
    assert output["realized_lgd_raw"].iloc[1] > 1
    assert output["realized_lgd_model_target"].between(0, 1).all()
    assert output["freddie_reconstructed_actual_loss"].iloc[0] == pytest.approx(-20.0)
    assert not set(config.features.numeric + config.features.categorical).intersection(
        OUTCOME_EXCLUDED_COLUMNS
    )


def test_reconciliation_formula_component_metadata_and_raw_lgd_preservation() -> None:
    config = load_lgd_config(Path.cwd())
    frame = pd.DataFrame(
        {
            "loan_id": ["A"],
            "resolved_flag": [True],
            "cured_flag": [False],
            "ead_at_default": [150.0],
            "ead_at_default_source": ["current_actual_upb"],
            "zero_balance_removal_upb": [100.0],
            "current_interest_rate_at_default": [0.0],
            "original_interest_rate": [0.0],
            "months_to_resolution": [0],
            "net_sale_proceeds": [-70.0],
            "mi_recoveries": [-5.0],
            "non_mi_recoveries": [-2.0],
            "total_expenses": [10.0],
            "delinquent_accrued_interest": [3.0],
            "cumulative_modification_costs": [4.0],
            "bankruptcy_cramdown_costs": [1.0],
            "actual_loss": [36.0],
        }
    )
    output = _add_lgd_targets(frame, config)
    reconciliation = _actual_loss_reconciliation(output)
    audit = _component_audit(output)

    assert output["freddie_reconstructed_actual_loss"].iloc[0] == pytest.approx(36.0)
    assert output["economic_loss"].iloc[0] == pytest.approx(41.0)
    assert output["realized_lgd_raw"].iloc[0] == pytest.approx(0.41)
    assert reconciliation["reconstructed_difference_sum"].iloc[0] == pytest.approx(0.0)
    assert audit.loc[
        audit["component"] == "cumulative_modification_costs",
        "component_type",
    ].iloc[0] == "CUMULATIVE"


def test_run_lgd_framework_persists_and_reloads(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo)

    result = run_lgd_framework(repo_root=repo, run_id="lgd_small", force=True)
    run = load_lgd_run(repo, "lgd_small")
    episodes = pd.read_parquet(Path(result.artifact_path) / "lgd_episodes.parquet")

    assert run["run_id"] == "lgd_small"
    assert Path(result.model_path, "lgd_models.pkl").exists()
    assert Path(result.artifact_path, "actual_loss_reconciliation.csv").exists()
    assert Path(result.artifact_path, "lgd_component_audit.csv").exists()
    assert Path(result.artifact_path, "lgd_distribution_diagnostics.csv").exists()
    assert Path(result.artifact_path, "component_decomposition.csv").exists()
    assert Path(result.artifact_path, "predictor_drift.csv").exists()
    assert Path(result.artifact_path, "cure_model_comparison.csv").exists()
    assert Path(result.artifact_path, "severity_model_comparison.csv").exists()
    assert Path(result.artifact_path, "combined_backtest.csv").exists()
    assert episodes["predicted_lgd"].between(0, 1).all()
    assert _validated_predictors(load_lgd_config(repo))
    with pytest.raises(FileExistsError):
        run_lgd_framework(repo_root=repo, run_id="lgd_small")


def test_elgd_branch_decomposition_reconciles() -> None:
    scored = pd.DataFrame(
        {
            "split": ["VALIDATION", "VALIDATION"],
            "resolved_flag": [True, True],
            "cured_flag": [True, False],
            "realized_lgd_model_target": [0.2, 0.8],
            "predicted_cure_probability": [0.75, 0.25],
            "predicted_cure_lgd": [0.3, 0.3],
            "predicted_non_cure_lgd": [0.7, 0.7],
            "predicted_lgd": [0.4, 0.6],
            "expected_lgd_calibration_factor": [1.0, 1.0],
        }
    )

    decomposition = _component_decomposition(scored)

    assert decomposition["predicted_branch_elgd"].iloc[0] == pytest.approx(0.5)
    assert decomposition["branch_reconciliation_difference"].iloc[0] == pytest.approx(0.0)


def test_probability_calibration_uses_supplied_validation_only_sample() -> None:
    probability = pd.Series([0.2, 0.3, 0.9, 0.95])
    validation_target = pd.Series([0, 0])
    oot_target = pd.Series([1, 1])

    validation_only = _fit_probability_recalibrator(probability.iloc[:2], validation_target)
    with_oot = _fit_probability_recalibrator(
        probability,
        pd.concat([validation_target, oot_target]),
    )

    assert validation_only["method"] == "none"
    assert with_oot["method"] == "validation_logistic_recalibration"


def test_temporal_weighting_prioritizes_recent_defaults() -> None:
    frame = pd.DataFrame(
        {
            "default_date": pd.to_datetime(
                ["2017-01-01", "2018-01-01", "2018-12-01"]
            )
        }
    )

    assert _temporal_training_weights(frame, "none", None) is None
    linear = _temporal_training_weights(frame, "linear_recency", None)
    exponential = _temporal_training_weights(frame, "exponential_recency", 12)

    assert linear[-1] > linear[0]
    assert exponential[-1] > exponential[0]
    assert linear.mean() == pytest.approx(1.0)
    assert exponential.mean() == pytest.approx(1.0)


def test_combined_selection_is_deterministic_and_excludes_oot() -> None:
    scored = pd.DataFrame(
        {
            "split": ["VALIDATION", "VALIDATION", "OOT", "OOT"],
            "resolved_flag": [True, True, True, True],
            "cured_flag": [False, False, False, False],
            "realized_lgd_model_target": [0.2, 0.2, 0.9, 0.9],
        }
    )
    cure_candidates = [
        {
            "name": "cure_a",
            "model": object(),
            "recalibrator": {"method": "none"},
            "raw_probability": pd.Series([0.0, 0.0, 0.0, 0.0]),
            "probability": np.array([0.0, 0.0, 0.0, 0.0]),
            "temporal_weighting": "none",
        },
        {
            "name": "cure_b",
            "model": object(),
            "recalibrator": {"method": "none"},
            "raw_probability": pd.Series([0.0, 0.0, 0.0, 0.0]),
            "probability": np.array([0.0, 0.0, 0.0, 0.0]),
            "temporal_weighting": "none",
        },
    ]
    severity_candidates = [
        {
            "name": "severity_validation_best",
            "model": object(),
            "calibrator": {"method": "none"},
            "prediction": pd.Series([0.2, 0.2, 0.2, 0.2]),
            "temporal_weighting": "none",
        },
        {
            "name": "severity_oot_best",
            "model": object(),
            "calibrator": {"method": "none"},
            "prediction": pd.Series([0.1, 0.3, 0.9, 0.9]),
            "temporal_weighting": "none",
        },
    ]
    cure_lgd = {"prediction": pd.Series([0.0, 0.0, 0.0, 0.0])}

    first = _combined_model_selection(scored, cure_candidates, severity_candidates, cure_lgd)
    second = _combined_model_selection(scored, cure_candidates, severity_candidates, cure_lgd)

    assert first[0]["severity_model_name"] == "severity_validation_best"
    assert first[0]["cure_model_name"] == second[0]["cure_model_name"]
    assert first[0]["severity_model_name"] == second[0]["severity_model_name"]
