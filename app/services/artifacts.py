"""Artifact loading helpers for the Streamlit portfolio app."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def repo_root() -> Path:
    """Return the repository root from the app package location."""
    return Path(__file__).resolve().parents[2]


def artifact_path(*parts: str) -> Path:
    """Build an artifact path relative to the repository root."""
    return repo_root().joinpath(*parts)


def require_path(path: Path) -> Path:
    """Raise a friendly missing-artifact error when a required path is absent."""
    if not path.exists():
        msg = f"Required artifact is missing: {path.relative_to(repo_root())}"
        raise FileNotFoundError(msg)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV artifact."""
    return pd.read_csv(require_path(path))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact."""
    with require_path(path).open() as stream:
        return json.load(stream)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet artifact."""
    return pd.read_parquet(require_path(path), columns=columns)


def discover_runs(category: str) -> list[str]:
    """Return available run IDs under an artifact category."""
    root = artifact_path("artifacts", category)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_scorecard_runs() -> list[str]:
    """Return available scorecard run IDs."""
    root = artifact_path("artifacts", "models", "scorecard")
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def git_commit() -> str:
    """Return the short git commit when available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def data_period() -> str:
    """Return the available Gold mart period when the manifest exists."""
    manifest = artifact_path("artifacts", "mart", "mart_manifest.json")
    if not manifest.exists():
        return "Unavailable"
    payload = read_json(manifest)
    start = payload.get("min_as_of_date") or payload.get("start_date")
    end = payload.get("max_as_of_date") or payload.get("end_date")
    if start and end:
        return f"{start} to {end}"
    return "See mart manifest"
