from __future__ import annotations

from pathlib import Path

import pandas as pd

from ifrs9.sicr.config import load_staging_config
from ifrs9.sicr.framework import allocate_stages, simulate_sicr_thresholds


def _base_frame() -> pd.DataFrame:
    false_flags = [False] * 9
    return pd.DataFrame(
        {
            "loan_id": ["A", "A", "A", "A", "B", "C", "D", "E", "F"],
            "as_of_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-03-01",
                ]
            ).date,
            "pd_current": [0.03, 0.012, 0.011, 0.010, 0.04, 0.01, 0.01, 0.01, 0.01],
            "rating_current": ["R3", "R2", "R2", "R2", "R5", "R1", "R1", "R1", "R1"],
            "ead_current": [100.0] * 9,
            "delinquency_months": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "ever_modified_to_date": false_flags,
            "ever_assistance_to_date": false_flags,
            "current_assistance_flag": false_flags,
            "origination_date": pd.to_datetime(["2019-01-01"] * 9).date,
            "default_entry_event": [False, False, False, False, True, False, False, False, False],
            "active_default_state": [False, False, False, False, True, False, False, False, False],
            "default_episode_id": [None, None, None, None, 1, None, None, None, None],
            "default_entry_date": pd.to_datetime(
                [
                    None,
                    None,
                    None,
                    None,
                    "2020-01-01",
                    None,
                    None,
                    None,
                    None,
                ]
            ).date,
            "cure_date": pd.to_datetime([None] * 9).date,
            "default_reason": [None, None, None, None, "DELINQUENCY", None, None, None, None],
            "sicr_reference_date": pd.to_datetime(["2020-01-01"] * 9).date,
            "sicr_reference_pd": [0.01] * 9,
            "sicr_reference_rating": ["R1"] * 9,
            "months_origination_to_reference": [12] * 9,
        }
    )


def test_stage3_precedence_and_core_triggers() -> None:
    config = load_staging_config(Path.cwd())
    staged = allocate_stages(_base_frame(), config)

    assert staged.loc[staged["loan_id"] == "B", "stage"].iloc[0] == 3
    assert staged.loc[staged["loan_id"] == "B", "stage_reason"].iloc[0] == "DEFAULT"
    assert staged.loc[staged["loan_id"] == "A", "sicr_relative_pd"].iloc[0]
    assert staged.loc[staged["loan_id"] == "B", "sicr_rating_downgrade"].iloc[0]
    assert staged.loc[staged["loan_id"] == "C", "sicr_dpd_backstop"].iloc[0]


def test_disabled_trigger_behavior() -> None:
    config = load_staging_config(Path.cwd()).model_copy(deep=True)
    config.sicr.relative_pd_increase = None
    frame = _base_frame().iloc[[0]].copy()

    staged = allocate_stages(frame, config)

    assert not staged["sicr_relative_pd"].iloc[0]


def test_reference_pd_no_future_leakage() -> None:
    config = load_staging_config(Path.cwd())
    staged = allocate_stages(_base_frame(), config)

    assert (staged["sicr_reference_date"] <= staged["as_of_date"]).iloc[0]
    assert staged["pd_origination"].iloc[0] == 0.01


def test_stage2_cure_probation_and_gaps_break_probation() -> None:
    config = load_staging_config(Path.cwd())
    staged = allocate_stages(_base_frame(), config)
    loan_a = staged[staged["loan_id"] == "A"].sort_values("as_of_date")
    loan_f = staged[staged["loan_id"] == "F"].sort_values("as_of_date")

    assert loan_a["stage"].tolist() == [2, 2, 2, 1]
    assert loan_f["stage"].iloc[0] == 1


def test_stage3_cure_behavior() -> None:
    config = load_staging_config(Path.cwd())
    frame = _base_frame()
    cure_rows = pd.DataFrame(
        {
            "loan_id": ["G", "G"],
            "as_of_date": pd.to_datetime(["2020-01-01", "2020-02-01"]).date,
            "pd_current": [0.01, 0.01],
            "rating_current": ["R1", "R1"],
            "ead_current": [100.0, 100.0],
            "delinquency_months": [3, 0],
            "ever_modified_to_date": [False, False],
            "ever_assistance_to_date": [False, False],
            "current_assistance_flag": [False, False],
            "origination_date": pd.to_datetime(["2019-01-01", "2019-01-01"]).date,
            "default_entry_event": [True, False],
            "active_default_state": [True, False],
            "default_episode_id": [1, None],
            "default_entry_date": pd.to_datetime(["2020-01-01", None]).date,
            "cure_date": pd.to_datetime(["2020-02-01", None]).date,
            "default_reason": ["DELINQUENCY", None],
            "sicr_reference_date": pd.to_datetime(["2020-01-01", "2020-01-01"]).date,
            "sicr_reference_pd": [0.01, 0.01],
            "sicr_reference_rating": ["R1", "R1"],
            "months_origination_to_reference": [12, 12],
        }
    )
    staged = allocate_stages(pd.concat([frame, cure_rows], ignore_index=True), config)
    loan_g = staged[staged["loan_id"] == "G"].sort_values("as_of_date")

    assert loan_g["stage"].tolist() == [3, 1]


def test_stage3_persists_until_cure_and_blocks_stage2() -> None:
    config = load_staging_config(Path.cwd())
    frame = pd.DataFrame(
        {
            "loan_id": ["H", "H", "H", "H"],
            "as_of_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ).date,
            "pd_current": [0.05, 0.05, 0.05, 0.01],
            "rating_current": ["R5", "R5", "R5", "R1"],
            "ead_current": [100.0, 95.0, 90.0, 85.0],
            "delinquency_months": [3, 2, 1, 0],
            "ever_modified_to_date": [False] * 4,
            "ever_assistance_to_date": [False] * 4,
            "current_assistance_flag": [False] * 4,
            "origination_date": pd.to_datetime(["2019-01-01"] * 4).date,
            "default_entry_event": [True, False, False, False],
            "active_default_state": [True, True, True, False],
            "default_episode_id": [1, 1, 1, None],
            "default_entry_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01", None]
            ).date,
            "cure_date": pd.to_datetime(["2020-04-01", "2020-04-01", "2020-04-01", None]).date,
            "default_reason": ["DELINQUENCY", "DELINQUENCY", "DELINQUENCY", None],
            "sicr_reference_date": pd.to_datetime(["2020-01-01"] * 4).date,
            "sicr_reference_pd": [0.01] * 4,
            "sicr_reference_rating": ["R1"] * 4,
            "months_origination_to_reference": [12] * 4,
        }
    )

    staged = allocate_stages(frame, config)

    assert staged["stage"].tolist()[:3] == [3, 3, 3]
    assert not staged.loc[staged["active_default_state"], "sicr_flag"].any()
    assert staged["stage"].tolist()[3] == 1


def test_stage3_redefault_starts_new_episode() -> None:
    config = load_staging_config(Path.cwd())
    frame = pd.DataFrame(
        {
            "loan_id": ["I", "I", "I", "I", "I"],
            "as_of_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
            ).date,
            "pd_current": [0.04, 0.04, 0.01, 0.04, 0.04],
            "rating_current": ["R4", "R4", "R1", "R4", "R4"],
            "ead_current": [100.0] * 5,
            "delinquency_months": [3, 2, 0, 3, 2],
            "ever_modified_to_date": [False] * 5,
            "ever_assistance_to_date": [False] * 5,
            "current_assistance_flag": [False] * 5,
            "origination_date": pd.to_datetime(["2019-01-01"] * 5).date,
            "default_entry_event": [True, False, False, True, False],
            "active_default_state": [True, True, False, True, True],
            "default_episode_id": [1, 1, None, 2, 2],
            "default_entry_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", None, "2020-04-01", "2020-04-01"]
            ).date,
            "cure_date": pd.to_datetime(
                ["2020-03-01", "2020-03-01", None, None, None]
            ).date,
            "default_reason": ["DELINQUENCY", "DELINQUENCY", None, "DELINQUENCY", "DELINQUENCY"],
            "sicr_reference_date": pd.to_datetime(["2020-01-01"] * 5).date,
            "sicr_reference_pd": [0.01] * 5,
            "sicr_reference_rating": ["R1"] * 5,
            "months_origination_to_reference": [12] * 5,
        }
    )

    staged = allocate_stages(frame, config)

    assert staged["stage"].tolist() == [3, 3, 1, 3, 3]
    assert staged["default_episode_id"].dropna().tolist() == [1, 1, 2, 2]


def test_unresolved_terminal_default_remains_stage3() -> None:
    config = load_staging_config(Path.cwd())
    frame = _base_frame().iloc[[4]].copy()
    frame["active_default_state"] = True
    frame["cure_date"] = pd.NaT

    staged = allocate_stages(frame, config)

    assert staged["stage"].iloc[0] == 3


def test_deterministic_sensitivity_simulation() -> None:
    config = load_staging_config(Path.cwd())
    staged = allocate_stages(_base_frame(), config)

    first = simulate_sicr_thresholds(
        config,
        staged,
        {
            "relative_pd_increase": [1.5, 2.0, 3.0],
            "rating_downgrade_notches": [3],
            "dpd_backstop_months": [1],
        },
    )
    second = simulate_sicr_thresholds(
        config,
        staged,
        {
            "relative_pd_increase": [1.5, 2.0, 3.0],
            "rating_downgrade_notches": [3],
            "dpd_backstop_months": [1],
        },
    )

    assert first.equals(second)
    assert first["stage2_count"].is_monotonic_decreasing
