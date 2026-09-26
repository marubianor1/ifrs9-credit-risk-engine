from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pandas as pd

from ifrs9.models.scorecard.runner import load_run, run_scorecard


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repo / "config" / "features").mkdir(parents=True)
    (repo / "config" / "models").mkdir(parents=True)
    shutil.copy(
        Path.cwd() / "config" / "features" / "point_in_time_features.yaml",
        repo / "config" / "features" / "point_in_time_features.yaml",
    )
    shutil.copy(
        Path.cwd() / "config" / "development_samples.yaml",
        repo / "config" / "development_samples.yaml",
    )
    scorecard_config = """
scorecard:
  model_type: logistic_scorecard
  population: behavioural
  snapshot_frequency: quarter_end
  sampling_strategy: none
  target: default_next_12m
  train:
    start: 2012-01-01
    end: 2018-12-01
  validation:
    start: 2019-01-01
    end: 2021-12-01
  oot:
    start: 2022-01-01
    end: 2025-03-01
  binning:
    max_bins: 4
    min_bin_pct: 0.05
    enforce_monotonic: true
    rare_category_pct: 0.05
    smoothing: 0.5
    suspicious_iv_threshold: 0.5
  score_scaling:
    base_score: 600
    pdo: 50
    base_odds_good_to_bad: 20
    band_count: 4
  metrics:
    confusion_cutoff: 0.4
    deciles: 4
  run:
    artifact_root: artifacts/models/scorecard
    model_root: models/scorecard
"""
    (repo / "config" / "models" / "scorecard.yaml").write_text(scorecard_config)
    (repo / "config" / "data.yaml").write_text(
        "\n".join(["ingestion:", "  gold_freddie_dir: data/gold/freddie", ""])
    )
    return repo


def _write_gold_and_targets(repo: Path) -> None:
    gold = repo / "data" / "gold" / "freddie"
    month = gold / "loan_month" / "vintage_year=2018"
    static = gold / "loan_static" / "vintage_year=2018"
    targets = gold / "targets" / "pd_12m_targets"
    month.mkdir(parents=True)
    static.mkdir(parents=True)
    targets.mkdir(parents=True)
    rows = []
    for split_year, start_idx in [(2018, 0), (2019, 40), (2022, 80)]:
        for index in range(40):
            loan_index = start_idx + index
            bad = 1 if index >= 28 else 0
            rows.append(
                (
                    f"L{loan_index:04d}",
                    2018,
                    f"{split_year}-03-01",
                    100000.0 - loan_index * 500.0,
                    float(bad * 3),
                    float(bad * 3),
                    float(bad),
                    1.0 - bad * 0.2,
                )
            )
    with duckdb.connect(database=":memory:") as con:
        con.register("month_rows", pd.DataFrame(rows, columns=[
            "loan_id",
            "vintage_year",
            "as_of_date",
            "current_actual_upb",
            "delinquency_months",
            "max_delinquency_months_to_date",
            "months_since_last_delinquency",
            "current_upb_to_original_upb",
        ]))
        con.execute(
            f"""
            COPY (
              SELECT
                  loan_id,
                  vintage_year,
                  CAST(as_of_date AS DATE) AS as_of_date,
                  current_actual_upb,
                  delinquency_months,
                  max_delinquency_months_to_date,
                  months_since_last_delinquency,
                  current_upb_to_original_upb
              FROM month_rows
            )
            TO '{(month / "part-loan_month.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT DISTINCT loan_id, vintage_year,
                     720.0 AS original_credit_score,
                     32.0 AS original_dti,
                     75.0 AS original_ltv,
                     75.0 AS original_cltv,
                     100000.0 AS original_upb,
                     6.0 AS original_interest_rate,
                     TIMESTAMP '2026-01-01 00:00:00' AS _gold_processed_at
              FROM month_rows
            )
            TO '{(static / "part-loan_static.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT loan_id, CAST(as_of_date AS DATE) AS as_of_date,
                     true AS eligible_for_12m_pd,
                     true AS target_12m_observable,
                     CASE WHEN delinquency_months > 0 THEN 1 ELSE 0 END AS default_next_12m,
                     false AS event_censored,
                     CASE WHEN delinquency_months > 0 THEN 'DEFAULT' ELSE 'CENSORED' END
                        AS event_type
              FROM month_rows
            )
            TO '{(targets / "part-pd_12m_targets.parquet").as_posix()}'
            (FORMAT PARQUET)
            """
        )


def test_scorecard_run_persists_artifacts_excludes_restricted_and_scores_direction(
    tmp_path: Path,
) -> None:
    repo = _repo_fixture(tmp_path)
    _write_gold_and_targets(repo)

    result = run_scorecard(repo_root=repo, population="behavioural", run_id="test_run")

    artifact = Path(result.artifact_path)
    run = load_run(repo, "test_run")
    features = pd.read_csv(artifact / "features.csv")["feature"].to_list()
    coefficients = pd.read_csv(artifact / "coefficients.csv")
    scored = pd.read_parquet(artifact / "scored_rows.parquet")
    metrics = pd.read_csv(artifact / "metrics.csv")

    assert run["run_id"] == "test_run"
    assert "actual_loss" not in features
    assert "_gold_processed_at" not in features
    assert (artifact / "binning.csv").exists()
    assert (Path(result.model_path) / "model.pkl").exists()
    assert set(metrics["split"]) == {"TRAIN", "VALIDATION", "OOT"}
    assert scored.loc[scored["target"] == 0, "score"].mean() > scored.loc[
        scored["target"] == 1,
        "score",
    ].mean()
    assert (coefficients["feature"] == "INTERCEPT").any()


def test_scorecard_run_is_deterministic_for_same_run_id(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    _write_gold_and_targets(repo)

    first = run_scorecard(repo_root=repo, population="behavioural", run_id="stable")
    first_metrics = pd.read_csv(Path(first.artifact_path) / "metrics.csv")
    second = run_scorecard(repo_root=repo, population="behavioural", run_id="stable")
    second_metrics = pd.read_csv(Path(second.artifact_path) / "metrics.csv")

    pd.testing.assert_frame_equal(first_metrics, second_metrics)
