from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from ifrs9.pd.config import PDFrameworkConfig, load_pd_config
from ifrs9.pd.framework import (
    _assign_ratings,
    _fit_calibration,
    _lifetime_curves,
    _rating_boundaries,
    _rating_summary,
    _transition_matrices,
    _transition_summary,
    load_pd_run,
    run_pd_framework,
)


def _config() -> PDFrameworkConfig:
    return load_pd_config(Path.cwd())


def _frame() -> pd.DataFrame:
    rows = []
    dates = {"TRAIN": "2018-03-01", "VALIDATION": "2019-03-01", "OOT": "2022-03-01"}
    for split, base in [("TRAIN", 0), ("VALIDATION", 40), ("OOT", 80)]:
        for index in range(40):
            pd_raw = 0.01 + index * 0.002
            target = int(index >= 30)
            rows.append(
                {
                    "loan_id": f"L{base + index:04d}",
                    "as_of_date": pd.Timestamp(dates[split]).date(),
                    "split": split,
                    "score": 700 - index,
                    "predicted_pd_raw": pd_raw,
                    "target": target,
                    "sample_weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_logistic_calibration_uses_validation_not_oot() -> None:
    config = _config()
    frame = _frame()
    model, calibrated = _fit_calibration(frame, config)
    modified_oot = frame.copy()
    modified_oot.loc[modified_oot["split"] == "OOT", "target"] = 1 - modified_oot.loc[
        modified_oot["split"] == "OOT",
        "target",
    ]
    changed_model, changed_calibrated = _fit_calibration(modified_oot, config)

    assert model == changed_model
    np.testing.assert_allclose(
        calibrated[frame["split"] != "OOT"],
        changed_calibrated[frame["split"] != "OOT"],
    )
    assert model["fit_split"] == "VALIDATION"


def test_ratings_are_monotonic_by_calibrated_pd() -> None:
    config = _config()
    frame = _frame()
    _, calibrated = _fit_calibration(frame, config)
    frame["pd_calibrated_12m"] = calibrated
    boundaries = _rating_boundaries(frame, config)
    frame["rating"] = _assign_ratings(frame, boundaries)
    summary = _rating_summary(frame, config)

    assert summary["mean_pd"].is_monotonic_increasing
    assert summary["rating"].iloc[0] == "R1"


def test_lifetime_curves_are_valid_and_reconcile_12_month_pd() -> None:
    config = _config()
    frame = pd.DataFrame(
        {
            "loan_id": [f"L{i:03d}" for i in range(60)],
            "rating": ["R1"] * 30 + ["R2"] * 30,
            "pd_calibrated_12m": [0.04] * 30 + [0.12] * 30,
            "sample_weight": [1.0] * 60,
        }
    )
    events = pd.DataFrame(
        {
            "loan_id": frame["loan_id"],
            "rating": frame["rating"],
            "months_until_event": [6] * 4 + [18] * 26 + [9] * 8 + [24] * 22,
            "event_default": [True] * 4 + [False] * 26 + [True] * 8 + [False] * 22,
            "event_prepayment_or_exit": [False] * 60,
            "event_censored": [False] * 60,
            "event_type": ["DEFAULT"] * 12 + ["CENSORED"] * 48,
        }
    )

    curves = _lifetime_curves(events, frame, config)
    for rating, group in curves.groupby("rating"):
        assert group["cumulative_pd"].between(0, 1).all()
        assert group["cumulative_pd"].is_monotonic_increasing
        assert group["survival_probability"].is_monotonic_decreasing
        target = frame.loc[frame["rating"] == rating, "pd_calibrated_12m"].mean()
        assert group.loc[group["month"] == 12, "cumulative_pd"].iloc[0] == pytest.approx(
            target,
            abs=0.01,
        )


def test_transition_matrices_sum_to_one_and_default_is_absorbing() -> None:
    config = _config()
    frame = pd.DataFrame(
        {
            "loan_id": ["A", "A", "B", "B", "C", "C"],
            "as_of_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]
            ).date,
            "rating": ["R1", "R2", "R2", "R1", "R3", "R3"],
            "target": [0, 0, 0, 0, 0, 1],
        }
    )
    matrices = _transition_matrices(frame, config)
    one_month = matrices[1]
    probabilities = one_month.groupby("rating_from")["probability"].sum()
    summary = _transition_summary(matrices, config)

    assert probabilities.loc["R1"] == pytest.approx(1.0)
    assert probabilities.loc["R2"] == pytest.approx(1.0)
    assert probabilities.loc["R3"] == pytest.approx(1.0)
    assert probabilities.loc["DEFAULT"] == pytest.approx(1.0)
    assert "default_transition_rate" in summary.columns


def _write_pd_repo(repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config").mkdir()
    (repo / "config" / "pd.yaml").write_text((Path.cwd() / "config" / "pd.yaml").read_text())
    scorecard = repo / "artifacts" / "models" / "scorecard" / "small_scorecard"
    scorecard.mkdir(parents=True)
    targets = repo / "data" / "gold" / "freddie" / "targets" / "pd_12m_targets"
    targets.mkdir(parents=True)
    rows = _frame()
    rows.to_parquet(scorecard / "scored_rows.parquet", index=False)
    target_rows = rows[
        [
            "loan_id",
            "as_of_date",
        ]
    ].copy()
    target_rows["months_until_event"] = np.where(rows["target"] == 1, 6, 24)
    target_rows["event_default"] = rows["target"].astype(bool)
    target_rows["event_prepayment_or_exit"] = False
    target_rows["event_censored"] = rows["target"] == 0
    target_rows["event_type"] = np.where(rows["target"] == 1, "DEFAULT", "CENSORED")
    with duckdb.connect(database=":memory:") as con:
        con.register("target_rows", target_rows)
        con.execute(
            f"""
            COPY target_rows
            TO '{(targets / "part-pd_12m_targets.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )


def test_run_pd_framework_persists_artifacts_and_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_pd_repo(repo)

    result = run_pd_framework(
        repo_root=repo,
        scorecard_run_id="small_scorecard",
        run_id="pd_small",
        force=True,
    )
    artifact = Path(result.artifact_path)
    run = load_pd_run(repo, "pd_small")

    assert run["scorecard_parent_run_id"] == "small_scorecard"
    assert (artifact / "pd_predictions.parquet").exists()
    assert (artifact / "rating_boundaries.csv").exists()
    assert (artifact / "lifetime_curves.csv").exists()
    assert (artifact / "transition_summary.csv").exists()
    assert (Path(result.model_path) / "calibration_model.pkl").exists()

    with pytest.raises(FileExistsError):
        run_pd_framework(repo_root=repo, scorecard_run_id="small_scorecard", run_id="pd_small")
