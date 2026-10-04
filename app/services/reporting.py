"""Reporting services for the Streamlit app."""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from app.services.artifacts import repo_root
from ifrs9.reporting import (
    REPORT_TYPES as BACKEND_REPORT_TYPES,
)
from ifrs9.reporting import (
    ReportGenerationResult,
    build_report_context,
    generate_report,
    load_reporting_config,
    render_report_html,
    render_report_markdown,
)

REPORT_TYPES = list(BACKEND_REPORT_TYPES)
AUDIENCES = ["Executive", "Risk Committee", "Model Validation", "Technical"]
DETAIL_LEVELS = ["Concise", "Standard", "Detailed"]
SCENARIO_RUNS = [
    "scenario_baseline_v1",
    "scenario_mild_deterioration_v1",
    "scenario_severe_deterioration_v1",
]


def groq_key_available() -> bool:
    """Return whether a Groq API key is configured."""
    return bool(_api_key())


def build_context_for_ui(
    *,
    report_type: str,
    audience: str,
    detail: str,
    scenario_run: str,
):
    """Build report context for the UI."""
    return build_report_context(
        repo_root=repo_root(),
        report_type=report_type,
        audience=audience,
        detail=detail,
        scenario_run=scenario_run,
    )


def generate_report_for_ui(
    *,
    report_type: str,
    audience: str,
    detail: str,
    scenario_run: str,
    use_llm: bool,
) -> tuple[ReportGenerationResult, dict[str, Any]]:
    """Generate a report from compact validated context."""
    root = repo_root()
    config = load_reporting_config(root)
    context = build_report_context(
        repo_root=root,
        report_type=report_type,
        audience=audience,
        detail=detail,
        scenario_run=scenario_run,
    )
    result = generate_report(
        context=context,
        config=config,
        api_key=_api_key(),
        use_llm=use_llm,
    )
    return result, context.model_dump()


def export_payloads(result: ReportGenerationResult) -> dict[str, str]:
    """Create report download payloads."""
    return {
        "markdown": render_report_markdown(result),
        "json": json.dumps(result.model_dump(), indent=2, default=str) + "\n",
        "html": render_report_html(result),
    }


def _api_key() -> str | None:
    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None
