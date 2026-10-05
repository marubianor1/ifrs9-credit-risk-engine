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
    max_output_tokens: int = 1200
    context_char_limit: int = 14_000
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


class AllowedNumericClaim(BaseModel):
    """Validated numeric metric available for report claim checking."""

    path: str
    value: float
    unit: str


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
    allowed_numbers: list[AllowedNumericClaim]
    request_profile: dict[str, Any] = Field(default_factory=dict)


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
    ecl_downturn = _read_csv(repo_root, "artifacts/ecl/ecl_v1/downturn_lgd_sensitivity.csv")
    staging = _read_csv(repo_root, "artifacts/sicr/sicr_v1_1/stage_distribution.csv")
    triggers = _read_csv(repo_root, "artifacts/sicr/sicr_v1_1/trigger_distribution.csv")
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
    scenario_stage = _read_csv(
        repo_root,
        f"artifacts/scenarios/{scenario_run}/stage_comparison.csv",
    )
    scenario_rating = _read_csv(
        repo_root,
        f"artifacts/scenarios/{scenario_run}/rating_comparison.csv",
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

    full_payload = {
        "portfolio": {
            "total_ead": total_ead,
            "weighted_ecl": weighted_ecl,
            "coverage_ratio": weighted_ecl / total_ead if total_ead else 0.0,
            "loan_count": int(ecl_stage["loans"].sum()),
        },
        "staging": {
            "stage_distribution": stage_records,
            "active_stage_distribution": _records(staging),
            "trigger_distribution": _records(triggers),
            "migration_summary": _records(migrations.head(8)),
        },
        "ecl": {
            "base_ecl": base_ecl,
            "upside_ecl": _scenario_value(ecl_scenario, "UPSIDE"),
            "downside_ecl": _scenario_value(ecl_scenario, "DOWNSIDE"),
            "weighted_ecl": weighted_ecl,
            "by_stage": stage_records,
            "by_rating": rating_records,
            "downturn_sensitivity": _records(ecl_downturn),
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
            "stage_comparison": _records(scenario_stage),
            "rating_comparison": _records(scenario_rating.head(5)),
        },
    }
    context_payload = _project_report_context(report_type, full_payload, scenario_run)
    allowed_numbers = _collect_numeric_claims(context_payload)
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
        request_profile={
            "report_context_profile": report_type,
            "included_sections": [
                key for key, value in context_payload.items() if value not in ({}, [], None)
            ],
        },
        **context_payload,
    )


def write_context_snapshot(context: ReportContext, output_path: Path) -> Path:
    """Persist report context JSON for audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context.model_dump_json(indent=2) + "\n")
    return output_path


def _project_report_context(
    report_type: str,
    payload: dict[str, dict[str, Any]],
    scenario_run: str,
) -> dict[str, dict[str, Any]]:
    empty = {
        "portfolio": {},
        "staging": {},
        "ecl": {},
        "pd": {},
        "lgd": {},
        "ead": {},
        "monitoring": {},
        "scenario": {},
    }
    if report_type == "Executive IFRS 9 Summary":
        projected = empty | {
            "portfolio": payload["portfolio"],
            "ecl": {
                "scenario_totals": _select_keys(
                    payload["ecl"],
                    ["base_ecl", "upside_ecl", "downside_ecl", "weighted_ecl"],
                ),
                "by_stage": _compact_records(
                    payload["ecl"]["by_stage"],
                    [
                        "stage",
                        "loans",
                        "total_ead",
                        "weighted_ecl",
                        "coverage_ratio",
                    ],
                ),
                "downturn_sensitivity": payload["ecl"]["downturn_sensitivity"],
                "main_rating_concentration": _top_records(
                    payload["ecl"]["by_rating"],
                    sort_key="total_ead",
                    limit=3,
                    fields=["rating", "loans", "total_ead", "weighted_ecl", "coverage_ratio"],
                ),
            },
        }
    elif report_type == "Provisioning Report":
        projected = empty | {
            "portfolio": payload["portfolio"],
            "ecl": {
                "by_stage": _compact_records(
                    payload["ecl"]["by_stage"],
                    ["stage", "loans", "total_ead", "weighted_ecl", "coverage_ratio"],
                ),
                "scenario_totals": _select_keys(
                    payload["ecl"],
                    ["base_ecl", "upside_ecl", "downside_ecl", "weighted_ecl"],
                ),
                "rating_ecl_summary": _top_records(
                    payload["ecl"]["by_rating"],
                    sort_key="weighted_ecl",
                    limit=5,
                    fields=["rating", "loans", "total_ead", "weighted_ecl", "coverage_ratio"],
                ),
                "downturn_sensitivity": payload["ecl"]["downturn_sensitivity"],
            },
        }
    elif report_type == "Scorecard / PD Model Report":
        projected = empty | {
            "pd": payload["pd"],
            "monitoring": {
                "scorecard_metrics": payload["monitoring"]["scorecard_metrics"],
                "max_scorecard_psi": payload["monitoring"]["max_scorecard_psi"],
                "threshold_label": payload["monitoring"]["threshold_label"],
            },
        }
    elif report_type == "LGD Model Review":
        projected = empty | {"lgd": payload["lgd"]}
    elif report_type == "EAD Model Review":
        projected = empty | {"ead": payload["ead"]}
    elif report_type == "SICR / Staging Report":
        projected = empty | {"staging": payload["staging"]}
    elif report_type == "Scenario Stress Report":
        projected = empty | {
            "scenario": {
                "run_id": payload["scenario"].get("run_id", scenario_run),
                "name": payload["scenario"].get("name"),
                "baseline_ecl": payload["scenario"].get("baseline_ecl"),
                "stressed_ecl": payload["scenario"].get("stressed_ecl"),
                "delta_ecl": payload["scenario"].get("delta_ecl"),
                "delta_ecl_pct": payload["scenario"].get("delta_ecl_pct"),
                "driver_waterfall": payload["scenario"]["driver_waterfall"],
                "stage_comparison": _compact_records(
                    payload["scenario"]["stage_comparison"],
                    [
                        "stage",
                        "baseline_ead",
                        "baseline_ecl",
                        "stressed_ead",
                        "stressed_ecl",
                        "delta_ecl",
                    ],
                ),
                "rating_comparison": _compact_records(
                    payload["scenario"]["rating_comparison"],
                    ["rating", "baseline_ecl", "stressed_ecl", "delta_ecl", "delta_ecl_pct"],
                ),
            },
        }
    elif report_type == "Model Monitoring Report":
        projected = empty | {"monitoring": payload["monitoring"]}
    else:
        projected = empty | {
            "portfolio": payload["portfolio"],
            "ecl": _select_keys(
                payload["ecl"],
                ["base_ecl", "upside_ecl", "downside_ecl", "weighted_ecl"],
            ),
        }
    return projected


def _select_keys(mapping: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def _compact_records(records: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in fields if field in row} for row in records]


def _top_records(
    records: list[dict[str, Any]],
    *,
    sort_key: str,
    limit: int,
    fields: list[str],
) -> list[dict[str, Any]]:
    rows = sorted(records, key=lambda row: float(row.get(sort_key) or 0.0), reverse=True)
    return _compact_records(rows[:limit], fields)


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


def _collect_numeric_claims(value: Any, path: str = "") -> list[AllowedNumericClaim]:
    claims: list[AllowedNumericClaim] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            claims.extend(_collect_numeric_claims(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            claims.extend(_collect_numeric_claims(item, f"{path}[{index}]"))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        claims.append(AllowedNumericClaim(path=path, value=float(value), unit=_numeric_unit(path)))
    return claims


def _numeric_unit(path: str) -> str:
    name = path.lower()
    if any(
        token in name
        for token in [
            "coverage_ratio",
            "oe_ratio",
            "cure_rate",
            "bad_rate",
            "ead_ratio",
            "mean_pd",
            "min_pd",
            "max_pd",
            "delta_ecl_pct",
            "mape",
            "psi",
            "gini",
            "ks",
            "auc",
            "brier",
            "rmse",
            "mae",
        ]
    ):
        return "percentage_ratio"
    if any(token in name for token in ["ead", "ecl"]):
        return "currency"
    if any(
        token in name
        for token in [
            "loans",
            "rows",
            "defaults",
            "population",
            "stage",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "observable_defaults",
        ]
    ):
        return "count"
    return "numeric"
