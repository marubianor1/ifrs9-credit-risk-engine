"""Structured report context assembly from validated aggregate artifacts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

REPORT_TYPES = [
    "Executive IFRS 9 Summary",
    "Provisioning Report",
    "Scorecard / PD Model Report",
    "LGD Model Review",
    "EAD Model Review",
    "SICR / Staging Report",
    "Scenario Stress Report",
    "Model Monitoring Report",
]


class ReportingConfig(BaseModel):
    """Configuration for AI-assisted reporting."""

    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    max_output_tokens: int = 4000
    temperature: float = 0.2
    reasoning_effort: str | None = None
    templates: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class SourceRuns(BaseModel):
    """Source artifact run identifiers."""

    scorecard: str
    pd: str
    lgd: str
    ead: str
    staging: str
    ecl: str
    scenario: str


class ReportContext(BaseModel):
    """Compact context passed to narrative generation."""

    report_type: str
    audience: str
    detail: str
    generated_at: str
    reporting_date: str
    source_runs: SourceRuns
    portfolio: dict[str, Any]
    staging: dict[str, Any]
    ecl: dict[str, Any]
    pd: dict[str, Any]
    lgd: dict[str, Any]
    ead: dict[str, Any]
    monitoring: dict[str, Any]
    scenario: dict[str, Any]
    limitations: list[str]
    allowed_numbers: list[str]


def load_reporting_config(repo_root: Path, path: Path | None = None) -> ReportingConfig:
    """Load reporting configuration."""
    config_path = path or repo_root / "config" / "reporting.yaml"
    with config_path.open() as stream:
        payload = yaml.safe_load(stream) or {}
    return ReportingConfig(**payload.get("reporting", {}))


def build_report_context(
    *,
    repo_root: Path,
    report_type: str,
    audience: str,
    detail: str,
    scenario_run: str = "scenario_mild_deterioration_v1",
    reporting_date: str = "2025-03-01",
) -> ReportContext:
    """Build compact reporting context from existing aggregate artifacts."""
    if report_type not in REPORT_TYPES:
        msg = f"Unsupported report type: {report_type}"
        raise ValueError(msg)

    ecl_stage = _read_csv(repo_root, "artifacts/ecl/ecl_v1/ecl_by_stage.csv")
    ecl_scenario = _read_csv(repo_root, "artifacts/ecl/ecl_v1/ecl_by_scenario.csv")
    ecl_rating = _read_csv(repo_root, "artifacts/ecl/ecl_v1/ecl_by_rating.csv")
    staging = _read_csv(repo_root, "artifacts/sicr/sicr_v1_1/stage_distribution.csv")
    migrations = _read_csv(repo_root, "artifacts/sicr/sicr_v1_1/stage_migrations.csv")
    pd_backtest = _read_csv(repo_root, "artifacts/pd/pd_behavioural_qe_v1/backtesting_split.csv")
    pd_rating = _read_csv(repo_root, "artifacts/pd/pd_behavioural_qe_v1/rating_summary.csv")
    lgd_backtest = _read_csv(repo_root, "artifacts/lgd/lgd_v1_2/backtesting_split.csv")
    lgd_downturn = _read_csv(repo_root, "artifacts/lgd/lgd_v1_2/downturn_overlay.csv")
    ead_backtest = _read_csv(repo_root, "artifacts/ead/ead_v1/backtest_by_split.csv")
    ead_ratio = _read_csv(repo_root, "artifacts/ead/ead_v1/ead_ratio_distribution.csv")
    scenario_summary = _read_json(repo_root, f"artifacts/scenarios/{scenario_run}/summary.json")
    scenario_waterfall = _read_csv(
        repo_root,
        f"artifacts/scenarios/{scenario_run}/driver_waterfall.csv",
    )
    scorecard = _read_csv(
        repo_root,
        "artifacts/models/scorecard/behavioural_qe_v1/metrics.csv",
    )
    scorecard_psi = _read_csv(
        repo_root,
        "artifacts/models/scorecard/behavioural_qe_v1/stability_psi.csv",
    )

    total_ead = float(ecl_stage["total_ead"].sum())
    weighted_ecl = _scenario_value(ecl_scenario, "WEIGHTED")
    base_ecl = _scenario_value(ecl_scenario, "BASE")
    stage_records = _records(ecl_stage)
    rating_records = _records(ecl_rating.head(10))

    context_payload = {
        "portfolio": {
            "total_ead": total_ead,
            "weighted_ecl": weighted_ecl,
            "coverage_ratio": weighted_ecl / total_ead if total_ead else 0.0,
            "loan_count": int(ecl_stage["loans"].sum()),
        },
        "staging": {
            "stage_distribution": stage_records,
            "active_stage_distribution": _records(staging),
            "migration_summary": _records(migrations.head(12)),
        },
        "ecl": {
            "base_ecl": base_ecl,
            "upside_ecl": _scenario_value(ecl_scenario, "UPSIDE"),
            "downside_ecl": _scenario_value(ecl_scenario, "DOWNSIDE"),
            "weighted_ecl": weighted_ecl,
            "by_stage": stage_records,
            "by_rating": rating_records,
        },
        "pd": {
            "run_id": "pd_behavioural_qe_v1",
            "backtesting_by_split": _records(pd_backtest),
            "rating_summary": _records(pd_rating.head(10)),
        },
        "lgd": {
            "run_id": "lgd_v1_2",
            "governance_status": "Production/project baseline",
            "rejected_challengers": ["lgd_v1_3", "lgd_fl_v1", "lgd_fl_v2"],
            "backtesting_by_split": _records(lgd_backtest),
            "downturn_factors": _records(lgd_downturn),
        },
        "ead": {
            "run_id": "ead_v1",
            "selected_method": "contractual_amortization",
            "ratio_distribution": _records(ead_ratio),
            "backtesting_by_split": _records(ead_backtest),
        },
        "monitoring": {
            "scorecard_metrics": _records(scorecard),
            "max_scorecard_psi": float(scorecard_psi["psi"].max()),
            "threshold_label": "Project monitoring thresholds, not regulatory limits",
        },
        "scenario": {
            "run_id": scenario_summary.get("run_id", scenario_run),
            "name": scenario_summary.get("scenario", scenario_run),
            "baseline_ecl": scenario_summary.get("baseline_ecl"),
            "stressed_ecl": scenario_summary.get("stressed_ecl"),
            "delta_ecl": scenario_summary.get("delta_ecl"),
            "delta_ecl_pct": scenario_summary.get("delta_ecl_pct"),
            "driver_waterfall": _records(scenario_waterfall),
        },
    }
    allowed_numbers = sorted(_collect_number_strings(context_payload))
    return ReportContext(
        report_type=report_type,
        audience=audience,
        detail=detail,
        generated_at=datetime.now(UTC).isoformat(),
        reporting_date=reporting_date,
        source_runs=SourceRuns(
            scorecard="behavioural_qe_v1",
            pd="pd_behavioural_qe_v1",
            lgd="lgd_v1_2",
            ead="ead_v1",
            staging="sicr_v1_1",
            ecl="ecl_v1",
            scenario=scenario_run,
        ),
        limitations=[
            "This is a portfolio/research implementation, not a regulatory production system.",
            (
                "Baseline LGD is structural and macro-neutral because macro LGD challengers "
                "were rejected."
            ),
            (
                "Stage 3 ECL uses a current-exposure times LGD approximation where detailed "
                "cashflows are unavailable."
            ),
            "EAD uses contractual amortization as the selected baseline for amortizing mortgages.",
            (
                "Narrative is AI-assisted commentary over validated artifacts and does not "
                "recalculate IFRS 9 metrics."
            ),
        ],
        allowed_numbers=allowed_numbers,
        **context_payload,
    )


def write_context_snapshot(context: ReportContext, output_path: Path) -> Path:
    """Persist report context JSON for audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context.model_dump_json(indent=2) + "\n")
    return output_path


def _read_csv(repo_root: Path, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(repo_root, relative_path))


def _read_json(repo_root: Path, relative_path: str) -> dict[str, Any]:
    with _artifact_path(repo_root, relative_path).open() as stream:
        return json.load(stream)


def _artifact_path(repo_root: Path, relative_path: str) -> Path:
    if os.getenv("IFRS9_APP_MODE", "local_full").strip().lower() != "cloud_demo":
        return repo_root / relative_path
    parts = Path(relative_path).parts
    if not parts or parts[0] != "artifacts":
        return repo_root / relative_path
    root = repo_root / "deployment"
    if len(parts) >= 3 and parts[1] == "models" and parts[2] == "scorecard":
        return root.joinpath("scoring", *parts[3:])
    mapping = {
        "ead": "ead",
        "ecl": "ecl",
        "lgd": "lgd",
        "pd": "pd",
        "scenarios": "scenarios",
        "sicr": "sicr",
    }
    target = mapping.get(parts[1])
    if target in {"ead", "ecl", "sicr"} and len(parts) >= 3:
        return root.joinpath(target, *parts[3:])
    if target is not None:
        return root.joinpath(target, *parts[2:])
    return repo_root / relative_path


def _scenario_value(frame: pd.DataFrame, scenario: str) -> float:
    return float(frame.loc[frame["scenario"].eq(scenario), "ecl"].iloc[0])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    sanitized = frame.where(pd.notna(frame), None)
    return sanitized.to_dict(orient="records")


def _collect_number_strings(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_collect_number_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_number_strings(item))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        found.update(_number_variants(float(value)))
    return found


def _number_variants(value: float) -> set[str]:
    variants = {
        str(int(value)) if value.is_integer() else str(value),
        f"{value:,.0f}",
        f"{value:,.1f}",
        f"{value:,.2f}",
        f"{value:.3f}",
        f"{value:.4f}",
    }
    if abs(value) < 1:
        variants.add(f"{value * 100:.2f}%")
    if abs(value) >= 1_000_000_000:
        variants.add(f"${value / 1_000_000_000:,.2f}B")
    if abs(value) >= 1_000_000:
        variants.add(f"${value / 1_000_000:,.2f}M")
    return variants
