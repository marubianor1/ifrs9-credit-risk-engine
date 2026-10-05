"""Reporting services for the Streamlit app."""

from __future__ import annotations

import json
import os
import time
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
from ifrs9.reporting.generator import groq_request_profile

REPORT_TYPES = list(BACKEND_REPORT_TYPES)
AUDIENCES = ["Executive", "Risk Committee", "Model Validation", "Technical"]
DETAIL_LEVELS = ["Concise", "Standard", "Detailed"]
SCENARIO_RUNS = [
    "scenario_baseline_v1",
    "scenario_mild_deterioration_v1",
    "scenario_severe_deterioration_v1",
]
REPORT_COOLDOWN_SECONDS = 30


def groq_key_available() -> bool:
    """Return whether a Groq API key is configured."""
    return _llm_reporting_enabled() and bool(_api_key())


def report_cooldown_remaining() -> int:
    """Return remaining per-session LLM cooldown seconds."""
    last_call = st.session_state.get("last_llm_report_call_at")
    if not last_call:
        return 0
    elapsed = time.time() - float(last_call)
    return max(0, int(REPORT_COOLDOWN_SECONDS - elapsed))


def mark_report_call() -> None:
    """Record a per-session LLM call timestamp."""
    st.session_state["last_llm_report_call_at"] = time.time()


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
    payload = context.model_dump()
    payload["request_profile"] = result.request_profile or groq_request_profile(context, config)
    return result, payload


def request_profile_for_ui(
    *,
    report_type: str,
    audience: str,
    detail: str,
    scenario_run: str,
) -> dict[str, Any]:
    """Return compact LLM request diagnostics without exposing prompt text."""
    root = repo_root()
    config = load_reporting_config(root)
    context = build_report_context(
        repo_root=root,
        report_type=report_type,
        audience=audience,
        detail=detail,
        scenario_run=scenario_run,
    )
    return groq_request_profile(context, config)


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


def _llm_reporting_enabled() -> bool:
    raw = os.getenv("ENABLE_LLM_REPORTING")
    if raw is None:
        try:
            raw = st.secrets.get("ENABLE_LLM_REPORTING")
        except Exception:
            raw = None
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
