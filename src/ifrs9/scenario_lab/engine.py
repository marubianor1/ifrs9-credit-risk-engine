"""Scenario Lab backend for IFRS 9 ECL sensitivity analysis."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ifrs9.ecl.config import ECLConfig, load_ecl_config
from ifrs9.ecl.engine import (
    _lgd_lookup,
    _pd_curves,
    _portfolio_lgd,
    _rating_lgd,
    _reporting_population,
    _scenario_ecl,
    _scenario_weights,
    _summary,
)
from ifrs9.ingestion.schema_registry import find_repo_root
from ifrs9.mart.views import register_gold_views
from ifrs9.scenario_lab.config import (
    EADShockConfig,
    LGDShockConfig,
    MacroShockConfig,
    PortfolioShockConfig,
    ScenarioConfig,
    ScenarioLabConfig,
    StagingShockConfig,
    load_scenario_lab_config,
)
from ifrs9.sicr.framework import RATING_NOTCH

SCENARIOS = ("UPSIDE", "BASE", "DOWNSIDE")
RATING_ORDER = ("R1", "R2", "R3", "R4", "R5")


@dataclass(frozen=True)
class ScenarioLabResult:
    """Result of a Scenario Lab run."""

    run_id: str
    preset: str
    artifact_path: str
    processing_time_seconds: float
    baseline_ead: float
    baseline_ecl: float
    stressed_ead: float
    stressed_ecl: float
    delta_ecl: float
    delta_ecl_pct: float


def run_scenario(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> ScenarioLabResult:
    """Run a Scenario Lab preset or override without refitting parent models."""
    start = time.perf_counter()
    root = find_repo_root(repo_root)
    lab_config = load_scenario_lab_config(root, config_path)
    scenario = _scenario_from_config(lab_config, preset, overrides)
    run_id = run_id or f"scenario_{scenario.name}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    artifact_dir = root / lab_config.output.artifact_root / run_id
    if not force and artifact_dir.exists():
        msg = f"Scenario Lab run already exists: {run_id}. Pass force=True to overwrite."
        raise FileExistsError(msg)
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ecl_config = load_ecl_config(root)
    baseline = _load_baseline(root, lab_config.parents.ecl)
    if _is_noop_scenario(scenario):
        stressed = baseline.copy()
        waterfall = _zero_waterfall(float(baseline["ecl_weighted"].sum()))
    else:
        state = _scenario_state(root, ecl_config)
        stressed_state = _apply_portfolio_shock(state, scenario.portfolio)
        stressed_state = _apply_staging_controls(stressed_state, scenario.staging)
        pd_curves, _ = _pd_curves(root, ecl_config)
        stressed_curves = _apply_macro_controls(pd_curves, scenario.macro, root, lab_config)
        stressed_state = _apply_lgd_controls(root, ecl_config, stressed_state, scenario.lgd)
        stressed_state = _apply_ead_controls(stressed_state, scenario.ead)
        stressed = _score_state(stressed_state, stressed_curves, ecl_config)
        waterfall = _driver_waterfall(root, ecl_config, state, baseline, scenario, lab_config)

    stage_comparison = _comparison(baseline, stressed, ["stage"])
    rating_comparison = _comparison(baseline, stressed, ["rating"])
    summary = _summary_payload(
        run_id,
        scenario,
        baseline,
        stressed,
        waterfall,
        time.perf_counter() - start,
    )

    _write_json(artifact_dir / "config.json", scenario.model_dump())
    _write_json(artifact_dir / "summary.json", summary)
    stage_comparison.to_csv(artifact_dir / "stage_comparison.csv", index=False)
    rating_comparison.to_csv(artifact_dir / "rating_comparison.csv", index=False)
    waterfall.to_csv(artifact_dir / "driver_waterfall.csv", index=False)

    return ScenarioLabResult(
        run_id=run_id,
        preset=scenario.name,
        artifact_path=str(artifact_dir),
        processing_time_seconds=time.perf_counter() - start,
        baseline_ead=summary["baseline_ead"],
        baseline_ecl=summary["baseline_ecl"],
        stressed_ead=summary["stressed_ead"],
        stressed_ecl=summary["stressed_ecl"],
        delta_ecl=summary["delta_ecl"],
        delta_ecl_pct=summary["delta_ecl_pct"],
    )


def _is_noop_scenario(scenario: ScenarioConfig) -> bool:
    return (
        scenario.macro.unemployment_shock == 0
        and scenario.macro.hpi_shock == 0
        and scenario.macro.gdp_shock == 0
        and scenario.macro.mortgage_rate_shock == 0
        and scenario.macro.rho_override is None
        and scenario.macro.scenario_weights is None
        and scenario.staging.relative_pd_threshold is None
        and scenario.staging.absolute_pd_threshold is None
        and scenario.staging.rating_downgrade_threshold is None
        and scenario.staging.dpd_backstop is None
        and scenario.staging.stage2_cure_probation is None
        and scenario.portfolio.rating_downgrade_share == 0
        and scenario.portfolio.delinquency_1m_share == 0
        and scenario.portfolio.delinquency_2m_share == 0
        and scenario.portfolio.delinquency_3m_default_share == 0
        and scenario.lgd.method == "structural_base"
        and scenario.lgd.manual_overlay_pp == 0
        and scenario.ead.multiplier == 1
    )


def _zero_waterfall(ecl: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"driver": "PD / macro effect", "starting_ecl": ecl, "ending_ecl": ecl, "effect": 0.0},
            {
                "driver": "Stage migration effect",
                "starting_ecl": ecl,
                "ending_ecl": ecl,
                "effect": 0.0,
            },
            {"driver": "LGD effect", "starting_ecl": ecl, "ending_ecl": ecl, "effect": 0.0},
            {"driver": "EAD effect", "starting_ecl": ecl, "ending_ecl": ecl, "effect": 0.0},
            {
                "driver": "interaction / residual",
                "starting_ecl": ecl,
                "ending_ecl": ecl,
                "effect": 0.0,
            },
        ]
    )


def compare_scenarios(
    run_ids: list[str],
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Compare persisted Scenario Lab run summaries."""
    root = find_repo_root(repo_root)
    lab_config = load_scenario_lab_config(root, config_path)
    rows = []
    for run_id in run_ids:
        path = root / lab_config.output.artifact_root / run_id / "summary.json"
        with path.open() as stream:
            payload = json.load(stream)
        rows.append(payload)
    return pd.DataFrame(rows)


def _scenario_from_config(
    config: ScenarioLabConfig,
    preset: str | None,
    overrides: dict[str, Any] | None,
) -> ScenarioConfig:
    key = preset or config.default_preset
    if key not in config.presets:
        msg = f"Unknown Scenario Lab preset: {key}"
        raise KeyError(msg)
    scenario = config.presets[key]
    if overrides:
        scenario = scenario.model_copy(update=overrides, deep=True)
    return scenario


def _scenario_state(repo_root: Path, config: ECLConfig) -> pd.DataFrame:
    population = _reporting_population(repo_root, config)
    staging = _staging_snapshot(repo_root, config.reporting_date.isoformat())
    state = population.merge(
        staging,
        on=["loan_id", "as_of_date"],
        how="left",
        suffixes=("", "_staging"),
    )
    lgd = _lgd_lookup(repo_root, config, population)
    state = state.merge(lgd, on=["loan_id", "default_episode_id"], how="left")
    state["lgd"] = state["lgd"].fillna(state["rating"].map(_rating_lgd(repo_root, config)))
    state["lgd"] = state["lgd"].fillna(_portfolio_lgd(repo_root, config)).clip(0, 1)
    state["base_stage"] = state["stage"]
    state["base_rating"] = state["rating"]
    state["base_ead"] = state["ead"]
    state["base_lgd"] = state["lgd"]
    return state.sort_values(["loan_id", "as_of_date"]).reset_index(drop=True)


def _staging_snapshot(repo_root: Path, reporting_date: str) -> pd.DataFrame:
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
                s.*,
                m.delinquency_months,
                m.current_interest_rate,
                COALESCE(
                    NULLIF(m.remaining_months_to_maturity_source, 0),
                    NULLIF(m.months_to_contractual_maturity, 0),
                    0
                ) AS remaining_months_to_maturity_reporting
            FROM staging s
            LEFT JOIN gold_loan_month m
              ON s.loan_id = m.loan_id
             AND s.as_of_date = m.as_of_date
            WHERE s.as_of_date = DATE '{reporting_date}'
            """
        ).fetchdf()


def _load_baseline(repo_root: Path, parent_ecl_run: str) -> pd.DataFrame:
    artifact = repo_root / "artifacts" / "ecl" / parent_ecl_run / "run.json"
    if not artifact.exists():
        msg = f"Missing parent ECL run artifact: {parent_ecl_run}"
        raise FileNotFoundError(msg)
    path = repo_root / "data" / "gold" / "freddie" / "ecl" / "part-ecl.parquet"
    baseline = pd.read_parquet(path)
    return baseline.copy()


def _apply_portfolio_shock(frame: pd.DataFrame, shock: PortfolioShockConfig) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(shock.seed)
    active = output.index[~output["stage"].eq(3).to_numpy()]
    downgrade_idx = _sample_indices(active, shock.rating_downgrade_share, rng)
    output.loc[downgrade_idx, "rating"] = _downgrade_ratings(output.loc[downgrade_idx, "rating"])
    output.loc[downgrade_idx, "rating_current"] = output.loc[downgrade_idx, "rating"]
    output["rating_notch_change"] = (
        output["rating"].map(RATING_NOTCH) - output["rating_origination"].map(RATING_NOTCH)
    )

    one_month = _sample_indices(active, shock.delinquency_1m_share, rng)
    two_month = _sample_indices(active, shock.delinquency_2m_share, rng)
    default_month = _sample_indices(active, shock.delinquency_3m_default_share, rng)
    output.loc[one_month, "delinquency_months"] = np.maximum(
        output.loc[one_month, "delinquency_months"].fillna(0),
        1,
    )
    output.loc[two_month, "delinquency_months"] = np.maximum(
        output.loc[two_month, "delinquency_months"].fillna(0),
        2,
    )
    output.loc[default_month, "delinquency_months"] = np.maximum(
        output.loc[default_month, "delinquency_months"].fillna(0),
        3,
    )
    output.loc[default_month, "active_default_state"] = True
    output["portfolio_rating_downgrade_shock"] = False
    output["portfolio_delinquency_shock"] = False
    output.loc[downgrade_idx, "portfolio_rating_downgrade_shock"] = True
    delinquency_idx = one_month.union(two_month).union(default_month)
    output.loc[delinquency_idx, "portfolio_delinquency_shock"] = True
    return output


def _sample_indices(index: pd.Index, share: float, rng: np.random.Generator) -> pd.Index:
    if share <= 0 or len(index) == 0:
        return pd.Index([])
    size = int(round(len(index) * share))
    if size <= 0:
        return pd.Index([])
    selected = rng.choice(index.to_numpy(), size=min(size, len(index)), replace=False)
    return pd.Index(selected)


def _downgrade_ratings(ratings: pd.Series) -> pd.Series:
    mapping = {"R1": "R2", "R2": "R3", "R3": "R4", "R4": "R5", "R5": "R5"}
    return ratings.map(mapping).fillna(ratings)


def _apply_staging_controls(frame: pd.DataFrame, shock: StagingShockConfig) -> pd.DataFrame:
    output = frame.copy()
    if (
        shock.relative_pd_threshold is None
        and shock.absolute_pd_threshold is None
        and shock.rating_downgrade_threshold is None
        and shock.dpd_backstop is None
        and shock.stage2_cure_probation is None
    ):
        output["stage3_flag"] = output["stage"].eq(3)
        return output
    rel_threshold = shock.relative_pd_threshold
    abs_threshold = shock.absolute_pd_threshold
    rating_threshold = shock.rating_downgrade_threshold
    dpd_threshold = shock.dpd_backstop
    relative = (
        output["origination_baseline_available"].fillna(False)
        & output["pd_relative_change"].ge(rel_threshold)
        if rel_threshold is not None
        else output["sicr_relative_pd"].fillna(False)
    )
    absolute = (
        output["origination_baseline_available"].fillna(False)
        & output["pd_absolute_change"].ge(abs_threshold)
        if abs_threshold is not None
        else output["sicr_absolute_pd"].fillna(False)
    )
    rating = (
        output["rating_notch_change"].ge(rating_threshold)
        if rating_threshold is not None
        else output["sicr_rating_downgrade"].fillna(False)
    )
    dpd = (
        output["delinquency_months"].fillna(0).ge(dpd_threshold)
        if dpd_threshold is not None
        else output["sicr_dpd_backstop"].fillna(False)
    )
    output["stage3_flag"] = output["active_default_state"].fillna(False).astype(bool)
    output["sicr_relative_pd"] = relative.fillna(False)
    output["sicr_absolute_pd"] = absolute.fillna(False)
    output["sicr_rating_downgrade"] = rating.fillna(False)
    output["sicr_dpd_backstop"] = dpd.fillna(False)
    output["sicr_flag"] = (
        output[
            [
                "sicr_relative_pd",
                "sicr_absolute_pd",
                "sicr_rating_downgrade",
                "sicr_dpd_backstop",
                "sicr_other_credit_deterioration",
            ]
        ]
        .fillna(False)
        .any(axis=1)
        & ~output["stage3_flag"]
    )
    if shock.stage2_cure_probation is not None and shock.stage2_cure_probation > 0:
        output["sicr_flag"] |= output["base_stage"].eq(2) & ~output["stage3_flag"]
    output["stage"] = np.select(
        [output["stage3_flag"], output["sicr_flag"]],
        [3, 2],
        default=1,
    )
    output["stage_reason"] = np.select(
        [output["stage3_flag"], output["sicr_flag"]],
        ["DEFAULT", "SCENARIO_SICR"],
        default="NO_SICR",
    )
    return output


def _apply_macro_controls(
    pd_curves: dict[tuple[str, str], pd.DataFrame],
    shock: MacroShockConfig,
    repo_root: Path,
    config: ScenarioLabConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    output = {key: curve.copy() for key, curve in pd_curves.items()}
    scale = _macro_pd_scale(shock, repo_root, config)
    for (scenario, _rating), curve in output.items():
        scenario_weight = (
            shock.scenario_weights.get(scenario)
            if shock.scenario_weights is not None and scenario in shock.scenario_weights
            else float(curve["scenario_weight"].iloc[0])
        )
        curve["marginal_pd"] = np.clip(curve["marginal_pd"] * scale, 0, 1)
        curve["scenario_weight"] = scenario_weight
    _scenario_weights(output)
    return output


def _macro_pd_scale(
    shock: MacroShockConfig,
    repo_root: Path,
    config: ScenarioLabConfig,
) -> float:
    base_rho = _parent_rho(repo_root, config.parents.pd)
    rho_effect = 0.0
    if shock.rho_override is not None and base_rho is not None and base_rho > 0:
        rho_effect = 0.50 * (shock.rho_override - base_rho) / base_rho
    scale = (
        1.0
        + 0.08 * shock.unemployment_shock
        - 0.010 * shock.hpi_shock
        - 0.040 * shock.gdp_shock
        + 0.030 * shock.mortgage_rate_shock
        + rho_effect
    )
    return float(max(scale, 0.01))


def _parent_rho(repo_root: Path, pd_run: str) -> float | None:
    path = repo_root / "artifacts" / "forward_looking" / pd_run / "run.json"
    if not path.exists():
        return None
    with path.open() as stream:
        payload = json.load(stream)
    rho = payload.get("rho")
    return float(rho) if rho is not None else None


def _apply_lgd_controls(
    repo_root: Path,
    ecl_config: ECLConfig,
    frame: pd.DataFrame,
    shock: LGDShockConfig,
) -> pd.DataFrame:
    output = frame.copy()
    config = ecl_config.model_copy(update={"lgd_method": shock.method})
    lookup = _lgd_lookup(repo_root, config, output)
    output = output.drop(columns=["lgd"], errors="ignore").merge(
        lookup,
        on=["loan_id", "default_episode_id"],
        how="left",
    )
    output["lgd"] = output["lgd"].fillna(output["rating"].map(_rating_lgd(repo_root, config)))
    output["lgd"] = output["lgd"].fillna(_portfolio_lgd(repo_root, config))
    output["lgd"] = (output["lgd"] + shock.manual_overlay_pp / 100.0).clip(0, 1)
    output["lgd_method"] = shock.method
    output["manual_lgd_overlay_pp"] = shock.manual_overlay_pp
    return output


def _apply_ead_controls(frame: pd.DataFrame, shock: EADShockConfig) -> pd.DataFrame:
    output = frame.copy()
    output["ead"] = (output["ead"] * shock.multiplier).clip(lower=0)
    output["ead_multiplier"] = shock.multiplier
    return output


def _score_state(
    frame: pd.DataFrame,
    pd_curves: dict[tuple[str, str], pd.DataFrame],
    config: ECLConfig,
) -> pd.DataFrame:
    scored = frame.copy()
    weights = _scenario_weights(pd_curves)
    for scenario in sorted(weights):
        scored[f"ecl_{scenario.lower()}"] = _scenario_ecl(scored, pd_curves, scenario, config)
    scored["ecl_weighted"] = sum(
        scored[f"ecl_{scenario.lower()}"] * weight for scenario, weight in weights.items()
    )
    scored["coverage_ratio"] = np.where(
        scored["ead"].gt(0),
        scored["ecl_weighted"] / scored["ead"],
        0,
    )
    scored["ecl_exceeds_ead"] = scored["ecl_weighted"] > scored["ead"]
    return scored


def _comparison(baseline: pd.DataFrame, stressed: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    base = _summary(baseline, groups).rename(
        columns={
            "loans": "baseline_loans",
            "total_ead": "baseline_ead",
            "weighted_ecl": "baseline_ecl",
            "ecl_upside": "baseline_ecl_upside",
            "ecl_base": "baseline_ecl_base",
            "ecl_downside": "baseline_ecl_downside",
            "ecl_exceeds_ead": "baseline_ecl_exceeds_ead",
            "coverage_ratio": "baseline_coverage_ratio",
        }
    )
    stress = _summary(stressed, groups).rename(
        columns={
            "loans": "stressed_loans",
            "total_ead": "stressed_ead",
            "weighted_ecl": "stressed_ecl",
            "ecl_upside": "stressed_ecl_upside",
            "ecl_base": "stressed_ecl_base",
            "ecl_downside": "stressed_ecl_downside",
            "ecl_exceeds_ead": "stressed_ecl_exceeds_ead",
            "coverage_ratio": "stressed_coverage_ratio",
        }
    )
    joined = base.merge(stress, on=groups, how="outer").fillna(0)
    joined["delta_ead"] = joined["stressed_ead"] - joined["baseline_ead"]
    joined["delta_ecl"] = joined["stressed_ecl"] - joined["baseline_ecl"]
    joined["delta_ecl_pct"] = np.where(
        joined["baseline_ecl"].ne(0),
        joined["delta_ecl"] / joined["baseline_ecl"],
        0,
    )
    return joined


def _driver_waterfall(
    repo_root: Path,
    ecl_config: ECLConfig,
    base_state: pd.DataFrame,
    baseline: pd.DataFrame,
    scenario: ScenarioConfig,
    lab_config: ScenarioLabConfig,
) -> pd.DataFrame:
    base_curves, _ = _pd_curves(repo_root, ecl_config)
    portfolio_state = _apply_portfolio_shock(base_state, scenario.portfolio)
    pd_curves = _apply_macro_controls(base_curves, scenario.macro, repo_root, lab_config)
    pd_only = _score_state(portfolio_state, pd_curves, ecl_config)
    staged = _apply_staging_controls(portfolio_state, scenario.staging)
    stage_step = _score_state(staged, pd_curves, ecl_config)
    lgd_step = _apply_lgd_controls(repo_root, ecl_config, staged, scenario.lgd)
    lgd_step = _score_state(lgd_step, pd_curves, ecl_config)
    ead_step = _apply_ead_controls(lgd_step, scenario.ead)
    final = _score_state(ead_step, pd_curves, ecl_config)
    totals = {
        "baseline": float(baseline["ecl_weighted"].sum()),
        "pd_macro": float(pd_only["ecl_weighted"].sum()),
        "stage": float(stage_step["ecl_weighted"].sum()),
        "lgd": float(lgd_step["ecl_weighted"].sum()),
        "ead": float(final["ecl_weighted"].sum()),
    }
    rows = [
        {
            "driver": "PD / macro effect",
            "starting_ecl": totals["baseline"],
            "ending_ecl": totals["pd_macro"],
            "effect": totals["pd_macro"] - totals["baseline"],
        },
        {
            "driver": "Stage migration effect",
            "starting_ecl": totals["pd_macro"],
            "ending_ecl": totals["stage"],
            "effect": totals["stage"] - totals["pd_macro"],
        },
        {
            "driver": "LGD effect",
            "starting_ecl": totals["stage"],
            "ending_ecl": totals["lgd"],
            "effect": totals["lgd"] - totals["stage"],
        },
        {
            "driver": "EAD effect",
            "starting_ecl": totals["lgd"],
            "ending_ecl": totals["ead"],
            "effect": totals["ead"] - totals["lgd"],
        },
    ]
    total_delta = totals["ead"] - totals["baseline"]
    residual = total_delta - sum(row["effect"] for row in rows)
    rows.append(
        {
            "driver": "interaction / residual",
            "starting_ecl": totals["ead"],
            "ending_ecl": totals["ead"] + residual,
            "effect": residual,
        }
    )
    return pd.DataFrame(rows)


def _summary_payload(
    run_id: str,
    scenario: ScenarioConfig,
    baseline: pd.DataFrame,
    stressed: pd.DataFrame,
    waterfall: pd.DataFrame,
    runtime: float,
) -> dict[str, Any]:
    baseline_ead = float(baseline["ead"].sum())
    baseline_ecl = float(baseline["ecl_weighted"].sum())
    stressed_ead = float(stressed["ead"].sum())
    stressed_ecl = float(stressed["ecl_weighted"].sum())
    delta = stressed_ecl - baseline_ecl
    return {
        "run_id": run_id,
        "scenario": scenario.name,
        "description": scenario.description,
        "baseline_ead": baseline_ead,
        "baseline_ecl": baseline_ecl,
        "stressed_ead": stressed_ead,
        "stressed_ecl": stressed_ecl,
        "delta_ecl": delta,
        "delta_ecl_pct": delta / baseline_ecl if baseline_ecl else 0.0,
        "waterfall_delta": float(waterfall["effect"].sum()),
        "processing_time_seconds": runtime,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_state(),
    }


def _git_state() -> dict[str, str]:
    repo_root = find_repo_root()
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


__all__ = ["ScenarioLabResult", "compare_scenarios", "run_scenario"]
