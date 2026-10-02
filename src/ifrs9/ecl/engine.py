"""IFRS 9 expected credit loss engine."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ifrs9.ead.framework import contractual_balance
from ifrs9.ecl.config import ECLConfig, load_ecl_config
from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.views import register_gold_views


@dataclass(frozen=True)
class ECLRunResult:
    """Result of an ECL engine run."""

    run_id: str
    ecl_path: str
    artifact_path: str
    processing_time_seconds: float


@dataclass(frozen=True)
class ECLScenarioResult:
    """Result of an ECL scenario simulation."""

    loan_level: pd.DataFrame
    stage_summary: pd.DataFrame
    scenario_summary: pd.DataFrame


def run_ecl(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> ECLRunResult:
    """Run the IFRS 9 ECL engine."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    config = load_ecl_config(root, config_path)
    run_id = run_id or f"ecl_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    ecl_dir = root / config.output.gold_path
    artifact_dir = root / config.output.artifact_root / run_id
    if not force and artifact_dir.exists():
        msg = f"ECL run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    if ecl_dir.exists() and force:
        shutil.rmtree(ecl_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ecl_dir.mkdir(parents=True, exist_ok=True)

    output = simulate_ecl(config, repo_root=root)
    loan_level = output.loan_level
    loan_level.to_parquet(ecl_dir / "part-ecl.parquet", index=False)
    output.stage_summary.to_csv(artifact_dir / "ecl_by_stage.csv", index=False)
    _summary(loan_level, ["rating"]).to_csv(artifact_dir / "ecl_by_rating.csv", index=False)
    _summary(loan_level, ["vintage_year"]).to_csv(artifact_dir / "ecl_by_vintage.csv", index=False)
    _summary(loan_level, ["property_state"]).to_csv(artifact_dir / "ecl_by_state.csv", index=False)
    output.scenario_summary.to_csv(artifact_dir / "ecl_by_scenario.csv", index=False)
    _scenario_reconciliation(loan_level).to_csv(
        artifact_dir / "scenario_reconciliation.csv",
        index=False,
    )
    _downturn_sensitivity(root, config, loan_level).to_csv(
        artifact_dir / "downturn_lgd_sensitivity.csv",
        index=False,
    )
    _explainability(loan_level).to_csv(artifact_dir / "ecl_explainability.csv", index=False)
    _alignment_warnings(root, config, loan_level).to_csv(
        artifact_dir / "alignment_warnings.csv",
        index=False,
    )
    _write_json(artifact_dir / "config_snapshot.json", config.model_dump())

    result = ECLRunResult(
        run_id=run_id,
        ecl_path=str(ecl_dir),
        artifact_path=str(artifact_dir),
        processing_time_seconds=time.perf_counter() - start,
    )
    manifest = {
        "run_id": run_id,
        "config": config.model_dump(),
        "git_commit": _git_state(root),
        "created_at": datetime.now(UTC).isoformat(),
        "result": asdict(result),
        "rows": len(loan_level),
        "total_ead": float(loan_level["ead"].sum()),
        "total_weighted_ecl": float(loan_level["ecl_weighted"].sum()),
    }
    _write_json(artifact_dir / "run.json", manifest)
    return result


def simulate_ecl(
    config: ECLConfig,
    overrides: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> ECLScenarioResult:
    """Simulate loan-level ECL using accepted parent artifacts."""
    root = find_repo_root(repo_root)
    if overrides:
        config = config.model_copy(update=overrides)
    population = _reporting_population(root, config)
    pd_curves, pd_scenario_anchor_date = _pd_curves(root, config)
    weights = _scenario_weights(pd_curves)
    lgd = _lgd_lookup(root, config, population)
    scored = population.merge(lgd, on=["loan_id", "default_episode_id"], how="left")
    scored["lgd"] = scored["lgd"].fillna(scored["rating"].map(_rating_lgd(root, config)))
    scored["lgd"] = scored["lgd"].fillna(_portfolio_lgd(root, config)).clip(0, 1)
    scored["pd_scenario_anchor_date"] = pd_scenario_anchor_date
    scored["pd_horizon_start_date"] = pd.Timestamp(config.reporting_date)

    scenario_columns = {}
    for scenario in sorted(weights):
        scenario_columns[scenario] = _scenario_ecl(scored, pd_curves, scenario, config)
        scored[f"ecl_{scenario.lower()}"] = scenario_columns[scenario]
    scored["ecl_weighted"] = sum(
        scored[f"ecl_{scenario.lower()}"] * weight for scenario, weight in weights.items()
    )
    scored["coverage_ratio"] = np.where(
        scored["ead"].gt(0),
        scored["ecl_weighted"] / scored["ead"],
        0,
    )
    scored["coverage_ratio"] = scored["coverage_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    scored["ecl_exceeds_ead"] = scored["ecl_weighted"] > scored["ead"]
    scored["stage_horizon_months"] = np.select(
        [scored["stage"].eq(1), scored["stage"].eq(2), scored["stage"].eq(3)],
        [
            np.minimum(scored["remaining_months_to_maturity"], 12),
            scored["remaining_months_to_maturity"],
            0,
        ],
        default=0,
    )
    output_columns = [
        "loan_id",
        "vintage_year",
        "property_state",
        "stage",
        "stage_reason",
        "rating",
        "ead",
        "lgd",
        "ecl_upside",
        "ecl_base",
        "ecl_downside",
        "ecl_weighted",
        "coverage_ratio",
        "remaining_months_to_maturity",
        "stage_horizon_months",
        "pd_horizon_start_date",
        "pd_scenario_anchor_date",
        "ecl_exceeds_ead",
    ]
    compact = scored[output_columns].copy()
    return ECLScenarioResult(
        loan_level=compact,
        stage_summary=_summary(compact, ["stage"]),
        scenario_summary=_scenario_summary(compact),
    )


def _reporting_population(repo_root: Path, config: ECLConfig) -> pd.DataFrame:
    staging_path = repo_root / "data" / "gold" / "freddie" / "staging" / "part-staging.parquet"
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        con.execute(
            f"""
            CREATE OR REPLACE VIEW staging AS
            SELECT * FROM read_parquet('{staging_path.as_posix()}')
            """
        )
        return con.execute(
            f"""
            SELECT
                s.loan_id,
                s.as_of_date,
                s.stage,
                s.stage_reason,
                s.rating_current AS rating,
                s.default_episode_id,
                s.ead_current AS ead,
                m.vintage_year,
                m.current_actual_upb,
                m.current_interest_rate,
                COALESCE(
                    NULLIF(m.remaining_months_to_maturity_source, 0),
                    NULLIF(m.months_to_contractual_maturity, 0),
                    0
                ) AS remaining_months_to_maturity,
                st.property_state
            FROM staging s
            LEFT JOIN gold_loan_month m
              ON s.loan_id = m.loan_id
             AND s.as_of_date = m.as_of_date
            LEFT JOIN gold_loan_static st
              ON m.loan_id = st.loan_id
             AND m.vintage_year = st.vintage_year
            WHERE s.as_of_date = DATE '{config.reporting_date.isoformat()}'
              AND COALESCE(s.ead_current, m.current_actual_upb, 0) > 0
            """
        ).fetchdf()


def _pd_curves(
    repo_root: Path,
    config: ECLConfig,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.Timestamp]:
    path = (
        repo_root
        / "artifacts"
        / "forward_looking"
        / config.pd_run
        / "scenario_lifetime_curves.csv"
    )
    curves = pd.read_csv(path)
    curves["date"] = pd.to_datetime(curves["date"])
    reporting = pd.Timestamp(config.reporting_date)
    available = curves.loc[curves["date"] >= reporting, "date"]
    curve_date = available.min() if not available.empty else curves["date"].max()
    selected = curves[curves["date"].eq(curve_date)].copy()
    output = {}
    for (scenario, rating), group in selected.groupby(["scenario", "rating"], observed=True):
        output[(str(scenario), str(rating))] = group.sort_values("month")
    return output, curve_date


def _scenario_weights(pd_curves: dict[tuple[str, str], pd.DataFrame]) -> dict[str, float]:
    rows = []
    for (scenario, _rating), curve in pd_curves.items():
        rows.append((scenario, float(curve["scenario_weight"].iloc[0])))
    weights = pd.DataFrame(rows, columns=["scenario", "weight"]).drop_duplicates()
    total = weights["weight"].sum()
    if abs(total - 1.0) > 1e-8:
        msg = f"Scenario weights must sum to 1.0, got {total:.8f}"
        raise ValueError(msg)
    return dict(zip(weights["scenario"], weights["weight"], strict=False))


def _lgd_lookup(repo_root: Path, config: ECLConfig, population: pd.DataFrame) -> pd.DataFrame:
    episodes = _load_lgd_episodes(repo_root, config)
    lgd_col = "lgd_downturn" if config.lgd_method == "downturn_sensitivity" else "lgd_base"
    return episodes[["loan_id", "default_episode_id", lgd_col]].rename(columns={lgd_col: "lgd"})


def _rating_lgd(repo_root: Path, config: ECLConfig) -> dict[str, float]:
    episodes = _load_lgd_episodes(repo_root, config)
    lgd_col = "lgd_downturn" if config.lgd_method == "downturn_sensitivity" else "lgd_base"
    return episodes.groupby("rating_at_default", observed=True)[lgd_col].mean().to_dict()


def _portfolio_lgd(repo_root: Path, config: ECLConfig) -> float:
    episodes = _load_lgd_episodes(repo_root, config)
    lgd_col = "lgd_downturn" if config.lgd_method == "downturn_sensitivity" else "lgd_base"
    return float(episodes[lgd_col].mean())


def _load_lgd_episodes(repo_root: Path, config: ECLConfig) -> pd.DataFrame:
    return pd.read_parquet(
        repo_root / "artifacts" / "lgd" / config.lgd_run / "lgd_episodes.parquet"
    )


def _scenario_ecl(
    frame: pd.DataFrame,
    pd_curves: dict[tuple[str, str], pd.DataFrame],
    scenario: str,
    config: ECLConfig,
) -> pd.Series:
    ecl = pd.Series(0.0, index=frame.index)
    stage3 = frame["stage"].eq(3)
    ecl.loc[stage3] = (frame.loc[stage3, "ead"] * frame.loc[stage3, "lgd"]).clip(lower=0)
    max_month = int(
        min(config.max_lifetime_months, max(frame["remaining_months_to_maturity"].max(), 12))
    )
    monthly_rate = frame["current_interest_rate"].fillna(0).clip(lower=0) / 100 / 12
    for month in range(1, max_month + 1):
        horizon = np.select(
            [frame["stage"].eq(1), frame["stage"].eq(2)],
            [12, frame["remaining_months_to_maturity"].clip(lower=0)],
            default=0,
        )
        active = horizon >= month
        if not active.any():
            continue
        ead_month = contractual_balance(
            frame.loc[active, "ead"],
            frame.loc[active, "current_interest_rate"].fillna(0),
            frame.loc[active, "remaining_months_to_maturity"].fillna(0),
            month,
        )
        discount = np.power(1 + monthly_rate.loc[active].to_numpy(dtype=float), -month)
        marginal = _marginal_pd_for_month(frame.loc[active, "rating"], pd_curves, scenario, month)
        ecl.loc[active] += (
            marginal * frame.loc[active, "lgd"].to_numpy(dtype=float) * ead_month * discount
        )
    return ecl.clip(lower=0)


def ecl_calendar_month(reporting_date: pd.Timestamp | str, horizon_month: int) -> pd.Timestamp:
    """Map an ECL horizon month to the prospective calendar month."""
    if horizon_month < 0:
        msg = "horizon_month must be non-negative"
        raise ValueError(msg)
    return pd.Timestamp(reporting_date) + pd.DateOffset(months=horizon_month)


def _marginal_pd_for_month(
    ratings: pd.Series,
    pd_curves: dict[tuple[str, str], pd.DataFrame],
    scenario: str,
    month: int,
) -> np.ndarray:
    values = np.zeros(len(ratings))
    for rating in ratings.dropna().astype(str).unique():
        curve = pd_curves.get((scenario, rating))
        if curve is None:
            continue
        row = curve[curve["month"].eq(month)]
        if row.empty:
            continue
        values[ratings.astype(str).to_numpy() == rating] = float(row["marginal_pd"].iloc[0])
    return values


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, dropna=False, observed=True)
        .agg(
            loans=("loan_id", "nunique"),
            total_ead=("ead", "sum"),
            weighted_ecl=("ecl_weighted", "sum"),
            ecl_upside=("ecl_upside", "sum"),
            ecl_base=("ecl_base", "sum"),
            ecl_downside=("ecl_downside", "sum"),
            ecl_exceeds_ead=("ecl_exceeds_ead", "sum"),
        )
        .reset_index()
        .assign(coverage_ratio=lambda x: x["weighted_ecl"] / x["total_ead"])
    )


def _scenario_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ["upside", "base", "downside"]:
        rows.append({"scenario": scenario.upper(), "ecl": frame[f"ecl_{scenario}"].sum()})
    rows.append({"scenario": "WEIGHTED", "ecl": frame["ecl_weighted"].sum()})
    return pd.DataFrame(rows)


def _scenario_reconciliation(frame: pd.DataFrame) -> pd.DataFrame:
    total_ead = frame["ead"].sum()
    return pd.DataFrame(
        [
            {
                "metric": "total_ead",
                "value": total_ead,
            },
            {"metric": "base_ecl", "value": frame["ecl_base"].sum()},
            {"metric": "weighted_ecl", "value": frame["ecl_weighted"].sum()},
            {"metric": "coverage_ratio", "value": frame["ecl_weighted"].sum() / total_ead},
            {"metric": "ecl_exceeds_ead_loans", "value": int(frame["ecl_exceeds_ead"].sum())},
        ]
    )


def _downturn_sensitivity(
    repo_root: Path,
    config: ECLConfig,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    sensitivity_config = config.model_copy(update={"lgd_method": "downturn_sensitivity"})
    sensitivity = simulate_ecl(sensitivity_config, repo_root=repo_root).loan_level
    return pd.DataFrame(
        [
            {
                "lgd_method": "structural_base",
                "weighted_ecl": baseline["ecl_weighted"].sum(),
                "coverage_ratio": baseline["ecl_weighted"].sum() / baseline["ead"].sum(),
            },
            {
                "lgd_method": "downturn_sensitivity",
                "weighted_ecl": sensitivity["ecl_weighted"].sum(),
                "coverage_ratio": sensitivity["ecl_weighted"].sum() / sensitivity["ead"].sum(),
            },
            {
                "lgd_method": "increment",
                "weighted_ecl": sensitivity["ecl_weighted"].sum() - baseline["ecl_weighted"].sum(),
                "coverage_ratio": (
                    sensitivity["ecl_weighted"].sum() - baseline["ecl_weighted"].sum()
                )
                / baseline["ead"].sum(),
            },
        ]
    )


def _explainability(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("stage", observed=True)
        .agg(
            loans=("loan_id", "nunique"),
            stage_allocation=("ead", "sum"),
            pd_contribution=("ecl_base", "sum"),
            lgd_contribution=("lgd", "mean"),
            ead_contribution=("ead", "sum"),
            discounting=("coverage_ratio", "mean"),
            scenario_weighting=("ecl_weighted", "sum"),
        )
        .reset_index()
    )


def _alignment_warnings(repo_root: Path, config: ECLConfig, frame: pd.DataFrame) -> pd.DataFrame:
    warnings = []
    anchor_date = (
        pd.to_datetime(frame["pd_scenario_anchor_date"].iloc[0])
        if "pd_scenario_anchor_date" in frame and not frame.empty
        else pd.NaT
    )
    if pd.notna(anchor_date) and anchor_date.date() != config.reporting_date:
        warnings.append(
            {
                "warning": "pd_scenario_anchor_after_reporting_date",
                "detail": (
                    f"Scenario anchor {anchor_date.date()} is the first prospective "
                    f"forward-looking macro node after reporting date {config.reporting_date}; "
                    "ECL horizon month 1 still starts one month after the reporting date."
                ),
                "rows": len(frame),
            }
        )
    warnings.append(
        {
            "warning": "missing_rating",
            "detail": "Rows use portfolio LGD fallback where rating is missing.",
            "rows": int(frame["rating"].isna().sum()),
        }
    )
    warnings.append(
        {
            "warning": "ecl_exceeds_ead",
            "detail": "Reported, not clipped.",
            "rows": int(frame["ecl_exceeds_ead"].sum()),
        }
    )
    return pd.DataFrame(warnings)


def _git_state(repo_root: Path) -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}
    return {"commit": commit, "dirty": str(bool(status))}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
