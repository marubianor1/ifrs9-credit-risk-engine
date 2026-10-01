from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ifrs9.ead.config import load_ead_config
from ifrs9.ead.framework import (
    EAD_OUTCOME_EXCLUDED_COLUMNS,
    _validated_predictors,
    build_ead_observation_dataset,
    contractual_balance,
    load_ead_run,
    run_ead_framework,
)


def _copy_config(repo: Path) -> None:
    (repo / "config").mkdir()
    shutil.copy(Path.cwd() / "config" / "ead.yaml", repo / "config" / "ead.yaml")


def _month_row(
    loan_id: str,
    date: str,
    *,
    current_upb: float,
    prior_upb: float | None = None,
    zero_balance_code: str | None = None,
) -> dict[str, object]:
    return {
        "loan_id": loan_id,
        "vintage_year": 2020,
        "as_of_date": pd.Timestamp(date).date(),
        "loan_age_source": 12,
        "months_on_book_derived": 12,
        "months_since_origination": 12,
        "remaining_months_to_maturity_source": 348,
        "months_to_contractual_maturity": 348,
        "current_actual_upb": current_upb,
        "current_upb_lag_1": prior_upb if prior_upb is not None else current_upb + 1000,
        "current_interest_rate": 6.0,
        "delinquency_months": 0 if zero_balance_code is None else 3,
        "delinquency_status_category": None,
        "delinquency_status_raw": None,
        "modification_flag": "N",
        "payment_deferral_flag": None,
        "borrower_assistance_status_code": None,
        "delinquency_due_to_disaster": False,
        "current_non_interest_bearing_upb": 0.0,
        "current_interest_bearing_upb": current_upb,
        "zero_balance_code": zero_balance_code,
        "zero_balance_effective_date": pd.Timestamp(date).date() if zero_balance_code else None,
        "defect_settlement_date": None,
        "due_date_of_last_paid_installment": None,
        "estimated_loan_to_value": 75.0,
        "mortgage_insurance_cancellation_indicator": None,
        "servicer_name": "TEST",
        "current_upb_to_original_upb": current_upb / 110_000 if current_upb else 0.0,
        "principal_reduction_to_date": 110_000 - current_upb,
        "interest_rate_change_from_origination": 0.5,
        "delinquency_months_lag_1": 0,
        "delinquency_months_lag_3": 0,
        "delinquency_months_lag_6": 0,
        "delinquency_months_lag_12": 0,
        "current_upb_lag_3": current_upb + 3000,
        "current_upb_lag_6": current_upb + 6000,
        "current_upb_lag_12": current_upb + 12000,
        "current_interest_rate_lag_1": 6.0,
        "current_interest_rate_lag_12": 6.0,
        "upb_change_1m": -1000.0,
        "upb_change_pct_1m": -0.01,
        "upb_change_3m": -3000.0,
        "upb_change_pct_3m": -0.03,
        "upb_change_6m": -6000.0,
        "upb_change_pct_6m": -0.06,
        "upb_change_12m": -12000.0,
        "upb_change_pct_12m": -0.12,
        "rate_change_1m": 0.0,
        "rate_change_12m": 0.0,
        "max_delinquency_months_to_date": 0 if zero_balance_code is None else 3,
        "months_delinquent_to_date": 0.0 if zero_balance_code is None else 3.0,
        "delinquency_event_current_month": False,
        "delinquency_events_to_date": 0,
        "months_since_last_delinquency": None,
        "max_delinquency_last_3m": 0,
        "max_delinquency_last_6m": 0,
        "max_delinquency_last_12m": 0,
        "months_delinquent_last_3m": 0,
        "months_delinquent_last_6m": 0,
        "months_delinquent_last_12m": 0,
        "months_1plus_delinquent_to_date": 0,
        "months_2plus_delinquent_to_date": 0,
        "months_3plus_delinquent_to_date": 0,
        "ever_modified_to_date": False,
        "months_since_last_modification": None,
        "ever_assistance_to_date": False,
        "current_assistance_flag": False,
        "observed_months_to_date": 12,
        "calendar_months_since_first_observation": 12,
        "has_reporting_gap_to_date": False,
        "_gold_processed_at": "test",
    }


def _static_row(loan_id: str) -> dict[str, object]:
    return {
        "loan_id": loan_id,
        "vintage_year": 2020,
        "origination_date": pd.Timestamp("2019-01-01").date(),
        "first_payment_date": pd.Timestamp("2019-02-01").date(),
        "maturity_date": pd.Timestamp("2049-01-01").date(),
        "original_credit_score": 700,
        "original_dti": 35,
        "original_ltv": 80,
        "original_cltv": 80,
        "original_upb": 110_000.0,
        "original_interest_rate": 5.5,
        "original_loan_term": 360,
        "occupancy_status": "P",
        "property_type": "SF",
        "loan_purpose": "P",
        "channel": "R",
        "mortgage_insurance_percentage": 0,
        "property_state": "CA",
        "_gold_processed_at": "test",
    }


def _write_repo(repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    _copy_config(repo)
    base = repo / "data" / "gold" / "freddie"
    (base / "targets" / "default_events").mkdir(parents=True)
    (base / "targets" / "pd_12m_targets").mkdir(parents=True)
    (base / "loan_month" / "vintage_year=2020").mkdir(parents=True)
    (base / "loan_static" / "vintage_year=2020").mkdir(parents=True)
    (repo / "artifacts" / "pd" / "pd_behavioural_qe_v1").mkdir(parents=True)

    months = []
    defaults = []
    targets = []
    static = []
    ratings = []
    for index, year in enumerate([2018, 2018, 2020, 2020, 2023, 2023], start=1):
        loan_id = f"D{index}"
        obs_date = f"{year}-01-01"
        default_date = f"{year}-04-01"
        months.append(_month_row(loan_id, obs_date, current_upb=100_000 - index * 1000))
        months.append(
            _month_row(
                loan_id,
                default_date,
                current_upb=0.0 if index == 1 else 95_000 - index * 1000,
                prior_upb=96_000.0 if index == 1 else None,
                zero_balance_code="09" if index == 1 else None,
            )
        )
        defaults.append(
            {
                "loan_id": loan_id,
                "vintage_year": 2020,
                "default_episode_id": 1,
                "default_entry_date": pd.Timestamp(default_date).date(),
                "cure_date": None,
                "default_reason": "DELINQUENCY_3_PLUS",
                "default_due_to_delinquency": True,
                "default_due_to_credit_event": False,
                "redefault_flag": False,
                "_target_processed_at": "test",
            }
        )
        targets.append(
            {
                "loan_id": loan_id,
                "vintage_year": 2020,
                "as_of_date": pd.Timestamp(obs_date).date(),
                "first_default_date": pd.Timestamp(default_date).date(),
                "eligible_for_lifetime_pd": True,
                "event_default": True,
                "event_prepayment_or_exit": False,
                "event_type": "DEFAULT",
                "months_to_default_from_observation": 3,
                "future_first_non_default_exit_date": None,
                "future_first_non_default_exit_reason": None,
            }
        )
        targets.append(
            {
                "loan_id": loan_id,
                "vintage_year": 2020,
                "as_of_date": pd.Timestamp(default_date).date(),
                "first_default_date": pd.Timestamp(default_date).date(),
                "eligible_for_lifetime_pd": False,
                "event_default": False,
                "event_prepayment_or_exit": False,
                "event_type": "CENSORED",
                "months_to_default_from_observation": None,
                "future_first_non_default_exit_date": None,
                "future_first_non_default_exit_reason": None,
            }
        )
        static.append(_static_row(loan_id))
        ratings.append(
            {
                "loan_id": loan_id,
                "as_of_date": pd.Timestamp(obs_date).date(),
                "rating": "R3",
            }
        )

    months.append(_month_row("P1", "2020-01-01", current_upb=90_000.0))
    months.append(_month_row("P1", "2020-03-01", current_upb=0.0, zero_balance_code="01"))
    targets.append(
        {
            "loan_id": "P1",
            "vintage_year": 2020,
            "as_of_date": pd.Timestamp("2020-01-01").date(),
            "first_default_date": None,
            "eligible_for_lifetime_pd": True,
            "event_default": False,
            "event_prepayment_or_exit": True,
            "event_type": "PREPAYMENT",
            "months_to_default_from_observation": None,
            "future_first_non_default_exit_date": pd.Timestamp("2020-03-01").date(),
            "future_first_non_default_exit_reason": "PREPAYMENT",
        }
    )
    static.append(_static_row("P1"))
    ratings.append(
        {
            "loan_id": "P1",
            "as_of_date": pd.Timestamp("2020-01-01").date(),
            "rating": "R2",
        }
    )

    pd.DataFrame(months).to_parquet(
        base / "loan_month" / "vintage_year=2020" / "part.parquet",
        index=False,
    )
    pd.DataFrame(static).to_parquet(
        base / "loan_static" / "vintage_year=2020" / "part.parquet",
        index=False,
    )
    pd.DataFrame(defaults).to_parquet(
        base / "targets" / "default_events" / "part.parquet",
        index=False,
    )
    pd.DataFrame(targets).to_parquet(
        base / "targets" / "pd_12m_targets" / "part.parquet",
        index=False,
    )
    pd.DataFrame(ratings).to_parquet(
        repo / "artifacts" / "pd" / "pd_behavioural_qe_v1" / "pd_predictions.parquet",
        index=False,
    )


def test_contractual_amortization_math_zero_interest_and_maturity() -> None:
    one_month = contractual_balance(100.0, 12.0, 2.0, 1.0)[()]
    zero_interest = contractual_balance(100.0, 0.0, 4.0, 1.0)[()]
    maturity = contractual_balance(100.0, 6.0, 2.0, 2.0)[()]

    assert one_month == pytest.approx(50.248756, rel=1e-5)
    assert zero_interest == pytest.approx(75.0)
    assert maturity == pytest.approx(0.0)


def test_ead_target_fallback_censoring_and_no_leakage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo)
    config = load_ead_config(repo)

    observations = build_ead_observation_dataset(repo, config)
    default_observations = observations[observations["event_type"] == "DEFAULT"]
    prepaid = observations[observations["loan_id"] == "P1"]

    assert default_observations["ead_at_default"].notna().all()
    fallback = default_observations[default_observations["loan_id"] == "D1"].iloc[0]
    assert fallback["ead_at_default"] == pytest.approx(96_000.0)
    assert fallback["ead_at_default_source"] == "prior_month_current_upb"
    assert prepaid["event_type"].iloc[0] == "PREPAYMENT"
    assert prepaid["event_type"].iloc[0] != "DEFAULT"
    assert not set(config.features.numeric + config.features.categorical).intersection(
        EAD_OUTCOME_EXCLUDED_COLUMNS
    )


def test_run_ead_framework_deterministic_outputs_and_reload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo)

    first = run_ead_framework(repo_root=repo, run_id="ead_small", force=True)
    first_backtest = pd.read_csv(Path(first.artifact_path) / "backtest_by_split.csv")
    run = load_ead_run(repo, "ead_small")
    second = run_ead_framework(repo_root=repo, run_id="ead_small_2", force=True)
    second_backtest = pd.read_csv(Path(second.artifact_path) / "backtest_by_split.csv")
    profiles = pd.read_parquet(Path(first.artifact_path) / "ead_profiles.parquet")

    assert run["run_id"] == "ead_small"
    assert Path(first.model_path, "ead_model.pkl").exists()
    assert first_backtest.equals(second_backtest)
    assert profiles[["ead_current_balance", "ead_contractual", "ead_modelled"]].ge(0).all().all()
    month_zero = profiles[profiles["month"] == 0]
    assert (
        month_zero["ead_current_balance"].round(6)
        == month_zero["ead_contractual"].round(6)
    ).all()
    assert _validated_predictors(load_ead_config(repo))
