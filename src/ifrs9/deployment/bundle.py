"""Build and audit compact public deployment artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROHIBITED_PATTERNS = [
    "bronze",
    "silver",
    "gold/freddie",
    "loan_month",
    "loan_static",
    "loan_id",
    "pd_predictions",
    "scored_rows",
    "ead_observations",
    "lgd_episodes",
    "part-ecl",
    "part-staging",
    "raw",
    ".zip",
    ".parquet",
    ".txt",
]


@dataclass(frozen=True)
class DeploymentBundleResult:
    """Deployment bundle audit result."""

    output_path: str
    file_count: int
    bundle_size_bytes: int
    largest_file: str
    largest_file_bytes: int
    loan_level_files_detected: int
    absolute_paths_detected: int
    secrets_detected: int


def build_deployment_bundle(
    repo_root: Path | None = None,
    force: bool = True,
) -> DeploymentBundleResult:
    """Build deterministic compact artifacts for Streamlit Community Cloud."""
    root = repo_root or _find_repo_root()
    output = root / "deployment"
    if output.exists() and force:
        shutil.rmtree(output)
    _create_dirs(output)

    _copy_portfolio(root, output)
    _copy_scorecard(root, output)
    _copy_tree_files(root / "artifacts/pd/pd_behavioural_qe_v1", output / "pd/pd_behavioural_qe_v1")
    _copy_lgd(root, output)
    _copy_ead(root, output)
    _copy_tree_files(root / "artifacts/sicr/sicr_v1_1", output / "sicr")
    _copy_tree_files(root / "artifacts/ecl/ecl_v1", output / "ecl")
    _copy_scenarios(root, output)
    _copy_monitoring(root, output)

    manifest = _manifest(root, output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = audit_deployment_bundle(output)
    if (
        audit.loan_level_files_detected
        or audit.absolute_paths_detected
        or audit.secrets_detected
    ):
        msg = (
            "Deployment safety audit failed: "
            f"loan_level={audit.loan_level_files_detected}, "
            f"absolute_paths={audit.absolute_paths_detected}, secrets={audit.secrets_detected}"
        )
        raise ValueError(msg)
    return audit


def audit_deployment_bundle(path: Path) -> DeploymentBundleResult:
    """Audit bundle for prohibited files, absolute paths, and secret-looking content."""
    files = sorted(item for item in path.rglob("*") if item.is_file())
    sizes = {item: item.stat().st_size for item in files}
    largest = max(files, key=lambda item: sizes[item], default=path)
    loan_level = [item for item in files if _is_prohibited_path(item.relative_to(path))]
    absolute_paths = []
    secrets = []
    for item in files:
        if item.suffix.lower() not in {".csv", ".json", ".md", ".toml", ".yaml"}:
            continue
        text = item.read_text(errors="ignore")
        if "/Users/" in text or "\\Users\\" in text:
            absolute_paths.append(item)
        if "gsk_" in text or "GROQ_API_KEY=" in text or "api_key=" in text:
            secrets.append(item)
    return DeploymentBundleResult(
        output_path=str(path),
        file_count=len(files),
        bundle_size_bytes=sum(sizes.values()),
        largest_file=str(largest.relative_to(path)) if largest != path else "",
        largest_file_bytes=sizes.get(largest, 0),
        loan_level_files_detected=len(loan_level),
        absolute_paths_detected=len(absolute_paths),
        secrets_detected=len(secrets),
    )


def _create_dirs(output: Path) -> None:
    for name in [
        "portfolio",
        "scoring",
        "pd",
        "lgd",
        "ead",
        "sicr",
        "ecl",
        "scenarios",
        "monitoring",
    ]:
        (output / name).mkdir(parents=True, exist_ok=True)


def _copy_portfolio(root: Path, output: Path) -> None:
    source = root / "artifacts/mart/mart_manifest.json"
    if source.exists():
        _copy_file(source, output / "portfolio/mart_manifest.json")


def _copy_scorecard(root: Path, output: Path) -> None:
    keep = {
        "binning.csv",
        "calibration_curve.csv",
        "coefficients.csv",
        "deciles.csv",
        "features.csv",
        "gains_lift.csv",
        "iv_ranking.csv",
        "metrics.csv",
        "run.json",
        "score_bands.csv",
        "stability_psi.csv",
        "yearly_performance.csv",
    }
    for run_dir in sorted((root / "artifacts/models/scorecard").glob("*")):
        if not run_dir.is_dir():
            continue
        for name in sorted(keep):
            source = run_dir / name
            if source.exists():
                _copy_file(source, output / "scoring" / run_dir.name / name)


def _copy_lgd(root: Path, output: Path) -> None:
    for run_id in ["lgd_v1_2", "lgd_v1_3"]:
        run_dir = root / "artifacts/lgd" / run_id
        _copy_tree_files(run_dir, output / "lgd" / run_id)
        episodes = run_dir / "lgd_episodes.parquet"
        if episodes.exists():
            frame = pd.read_parquet(episodes, columns=["resolved_flag", "cured_flag"])
            summary = pd.DataFrame(
                [
                    {
                        "episodes": len(frame),
                        "resolved": int(frame["resolved_flag"].fillna(False).sum()),
                        "cure_rate": float(frame["cured_flag"].fillna(False).mean())
                        if len(frame)
                        else 0.0,
                    }
                ]
            )
            summary.to_csv(output / "lgd" / run_id / "population_summary.csv", index=False)
    for run_id in ["lgd_fl_v1", "lgd_fl_v2"]:
        _copy_tree_files(
            root / "artifacts/lgd_forward_looking" / run_id,
            output / "lgd/forward_looking" / run_id,
        )


def _copy_ead(root: Path, output: Path) -> None:
    source_root = root / "artifacts/ead/ead_v1"
    _copy_tree_files(source_root, output / "ead")
    profiles = source_root / "ead_profiles.parquet"
    if profiles.exists():
        frame = pd.read_parquet(profiles)
        value_cols = [column for column in frame.columns if column.startswith("ead_")]
        profile = frame.groupby("month", observed=True)[value_cols].mean().reset_index()
        profile.to_csv(output / "ead/ead_profiles.csv", index=False)


def _copy_scenarios(root: Path, output: Path) -> None:
    for run_id in [
        "scenario_baseline_v1",
        "scenario_mild_deterioration_v1",
        "scenario_severe_deterioration_v1",
    ]:
        _copy_tree_files(root / "artifacts/scenarios" / run_id, output / "scenarios" / run_id)


def _copy_monitoring(root: Path, output: Path) -> None:
    for relative in [
        "artifacts/models/scorecard/behavioural_qe_v1/metrics.csv",
        "artifacts/models/scorecard/behavioural_qe_v1/stability_psi.csv",
        "artifacts/models/scorecard/behavioural_qe_v1/yearly_performance.csv",
        "artifacts/pd/pd_behavioural_qe_v1/calibration_metrics.csv",
        "artifacts/pd/pd_behavioural_qe_v1/backtesting_split.csv",
        "artifacts/pd/pd_behavioural_qe_v1/backtesting_rating_year.csv",
        "artifacts/lgd/lgd_v1_2/backtesting_split.csv",
        "artifacts/lgd/lgd_v1_2/component_decomposition.csv",
        "artifacts/ead/ead_v1/backtest_by_split.csv",
    ]:
        source = root / relative
        _copy_file(source, output / "monitoring" / source.name)


def _copy_tree_files(source_root: Path, target_root: Path) -> None:
    if not source_root.exists():
        return
    for source in sorted(source_root.iterdir()):
        if source.is_file() and source.suffix.lower() in {".csv", ".json"}:
            _copy_file(source, target_root / source.name)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        text = source.read_text()
        text = text.replace(str(_find_repo_root()), "<repo_root>")
        target.write_text(text)
        return
    shutil.copy2(source, target)


def _manifest(root: Path, output: Path) -> dict[str, Any]:
    return {
        "bundle_version": 1,
        "app_mode": "cloud_demo",
        "git_commit": _git_commit(root),
        "source_runs": {
            "scorecard": ["behavioural_qe_v1", "application_qe_v1"],
            "pd": "pd_behavioural_qe_v1",
            "lgd": ["lgd_v1_2", "lgd_v1_3"],
            "ead": "ead_v1",
            "sicr": "sicr_v1_1",
            "ecl": "ecl_v1",
            "scenarios": [
                "scenario_baseline_v1",
                "scenario_mild_deterioration_v1",
                "scenario_severe_deterioration_v1",
            ],
        },
        "excluded_data": [
            "Freddie Mac ZIP/TXT source files",
            "Bronze/Silver/Gold loan-level data",
            "loan_static and loan_month marts",
            "loan-level targets, staging, ECL, and reporting snapshots",
            "loan-level model predictions and scored rows",
        ],
        "file_count": len([item for item in output.rglob("*") if item.is_file()]),
    }


def _is_prohibited_path(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return any(pattern in normalized for pattern in PROHIBITED_PATTERNS)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
