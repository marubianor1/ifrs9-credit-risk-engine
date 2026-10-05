"""AI-assisted report generation with deterministic fallback."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ifrs9.reporting.context import ReportContext, ReportingConfig


class GeneratedReport(BaseModel):
    """Structured narrative returned by the reporting layer."""

    model_config = ConfigDict(extra="forbid")

    title: str
    executive_summary: str | None
    portfolio_position: str | None
    key_risk_movements: str | None
    model_performance: str | None
    scenario_analysis: str | None
    limitations: str | None
    management_actions: str | None


class ReportGenerationResult(BaseModel):
    """Report generation result with validation metadata."""

    model_config = ConfigDict(extra="forbid")

    report: GeneratedReport
    generated_at: str
    source: str
    source_runs: dict[str, str]
    unsupported_numbers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    request_profile: dict[str, Any] = Field(default_factory=dict)


GROQ_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "executive_summary": {"type": ["string", "null"]},
        "portfolio_position": {"type": ["string", "null"]},
        "key_risk_movements": {"type": ["string", "null"]},
        "model_performance": {"type": ["string", "null"]},
        "scenario_analysis": {"type": ["string", "null"]},
        "limitations": {"type": ["string", "null"]},
        "management_actions": {"type": ["string", "null"]},
    },
    "required": [
        "title",
        "executive_summary",
        "portfolio_position",
        "key_risk_movements",
        "model_performance",
        "scenario_analysis",
        "limitations",
        "management_actions",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Narrate only supplied data. Never calculate IFRS 9 values or invent numbers. "
    "Use null for unavailable sections. Return JSON matching the strict schema."
)


def generate_report(
    *,
    context: ReportContext,
    config: ReportingConfig,
    api_key: str | None = None,
    use_llm: bool = True,
) -> ReportGenerationResult:
    """Generate a structured report from validated context only."""
    request_profile = groq_request_profile(context, config)
    if not use_llm:
        return _fallback_result(context, "LLM disabled by user selection.", request_profile)
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        return _fallback_result(context, "GROQ_API_KEY is not configured.", request_profile)
    if config.provider.lower() != "groq":
        return _fallback_result(
            context,
            f"Unsupported reporting provider: {config.provider}.",
            request_profile,
        )
    try:
        report = _generate_with_groq(context, config, resolved_key)
    except Exception as exc:
        return _fallback_result(context, f"Groq generation failed: {exc}", request_profile)

    unsupported = validate_report_numbers(report, context)
    warnings = []
    if unsupported:
        warnings.append("Unsupported numerical claims were detected and should be reviewed.")
    return ReportGenerationResult(
        report=report,
        generated_at=datetime.now(UTC).isoformat(),
        source="groq_structured_output",
        source_runs=context.source_runs.model_dump(),
        unsupported_numbers=unsupported,
        warnings=warnings,
        request_profile=request_profile,
    )


def validate_report_numbers(report: GeneratedReport, context: ReportContext) -> list[str]:
    """Return numerical references that are not traceable to report context."""
    text = "\n".join(str(value) for value in report.model_dump().values())
    allowed = set(context.allowed_numbers)
    unsupported = []
    for number in _extract_numbers(text):
        if _is_ignorable_number(number):
            continue
        if number not in allowed and _normalized_number(number) not in allowed:
            unsupported.append(number)
    return sorted(set(unsupported))


def render_report_markdown(result: ReportGenerationResult) -> str:
    """Render a generated report as Markdown."""
    report = result.report
    sections = [
        ("Executive Summary", report.executive_summary),
        ("Portfolio Position", report.portfolio_position),
        ("Key Risk Movements", report.key_risk_movements),
        ("Model Performance", report.model_performance),
        ("Scenario Analysis", report.scenario_analysis),
        ("Limitations", report.limitations),
        ("Management Actions", report.management_actions),
    ]
    lines = [
        f"# {report.title}",
        "",
        f"Generated at: `{result.generated_at}`",
        "",
        "AI-assisted commentary over validated IFRS 9 artifacts.",
        "",
    ]
    for heading, body in sections:
        if body:
            lines.extend([f"## {heading}", "", body, ""])
    if result.warnings:
        lines.extend(
            [
                "## Validation Warnings",
                "",
                "\n".join(f"- {item}" for item in result.warnings),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_report_html(result: ReportGenerationResult) -> str:
    """Render a lightweight standalone HTML report."""
    markdown = render_report_markdown(result)
    body = "\n".join(
        f"<p>{line}</p>" if line and not line.startswith("#") else _heading_to_html(line)
        for line in markdown.splitlines()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>IFRS 9 Report</title></head><body>"
        f"{body}</body></html>"
    )


def _generate_with_groq(
    context: ReportContext,
    config: ReportingConfig,
    api_key: str,
) -> GeneratedReport:
    from groq import Groq

    client = Groq(api_key=api_key)
    messages, _ = groq_messages(context, config)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_completion_tokens": config.max_output_tokens,
        "response_format": groq_response_format(),
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    response = client.chat.completions.create(**kwargs)
    payload = response.choices[0].message.content
    if payload is None:
        msg = "Groq response did not contain message content."
        raise ValueError(msg)
    try:
        return GeneratedReport.model_validate_json(payload)
    except ValidationError:
        return GeneratedReport.model_validate(json.loads(payload))


def groq_messages(
    context: ReportContext,
    config: ReportingConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return compact Groq messages and request-size diagnostics."""
    payload = _prompt_payload(context)
    user_prompt = _user_prompt(payload)
    profile = _request_profile(context, user_prompt, config)
    if profile["context_char_count"] > config.context_char_limit:
        payload = _compact_prompt_payload(payload, list_limit=3)
        user_prompt = _user_prompt(payload)
        profile = _request_profile(context, user_prompt, config)
    if profile["context_char_count"] > config.context_char_limit:
        payload = _compact_prompt_payload(payload, list_limit=1)
        user_prompt = _user_prompt(payload)
        profile = _request_profile(context, user_prompt, config)
    if profile["context_char_count"] > config.context_char_limit:
        msg = (
            f"Report context exceeds configured Groq budget: "
            f"{profile['context_char_count']} chars > {config.context_char_limit} chars."
        )
        raise ValueError(msg)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], profile


def groq_request_profile(context: ReportContext, config: ReportingConfig) -> dict[str, Any]:
    """Return request diagnostics without exposing prompt text."""
    _, profile = groq_messages(context, config)
    return profile


def groq_response_format() -> dict[str, Any]:
    """Return the exact strict response_format payload sent to Groq."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ifrs9_report",
            "strict": True,
            "schema": GROQ_REPORT_SCHEMA,
        },
    }


def _fallback_result(
    context: ReportContext,
    reason: str,
    request_profile: dict[str, Any] | None = None,
) -> ReportGenerationResult:
    report = _fallback_report(context)
    return ReportGenerationResult(
        report=report,
        generated_at=datetime.now(UTC).isoformat(),
        source="deterministic_template",
        source_runs=context.source_runs.model_dump(),
        warnings=[reason, "LLM reporting unavailable. Quantitative pages remain functional."],
        request_profile=request_profile or {},
    )


def _fallback_report(context: ReportContext) -> GeneratedReport:
    portfolio = context.portfolio
    ecl = context.ecl
    scenario = context.scenario
    total_ead = float(portfolio.get("total_ead") or 0.0)
    weighted_ecl = float(portfolio.get("weighted_ecl") or ecl.get("weighted_ecl") or 0.0)
    coverage_ratio = float(portfolio.get("coverage_ratio") or 0.0)
    scenario_totals = ecl.get("scenario_totals", ecl)
    base_ecl = float(scenario_totals.get("base_ecl") or ecl.get("base_ecl") or 0.0)
    upside_ecl = float(scenario_totals.get("upside_ecl") or ecl.get("upside_ecl") or 0.0)
    downside_ecl = float(scenario_totals.get("downside_ecl") or ecl.get("downside_ecl") or 0.0)
    delta_ecl = float(scenario.get("delta_ecl") or 0.0)
    return GeneratedReport(
        title=f"{context.report_type} - {context.reporting_date}",
        executive_summary=(
            f"Portfolio weighted ECL is ${weighted_ecl / 1_000_000:,.2f}M "
            f"on total EAD of ${total_ead / 1_000_000_000:,.2f}B, with "
            f"coverage of {coverage_ratio * 100:.2f}%."
        ),
        portfolio_position=(
            f"Stage 1, Stage 2, and Stage 3 results are sourced from "
            f"{context.source_runs.ecl} and {context.source_runs.staging}."
        ),
        key_risk_movements=(
            f"The selected scenario run {scenario.get('run_id', context.source_runs.scenario)} "
            f"changes ECL by ${delta_ecl / 1_000_000:,.2f}M versus baseline."
        ),
        model_performance=(
            f"PD, LGD, and EAD diagnostics are sourced from {context.source_runs.pd}, "
            f"{context.source_runs.lgd}, and {context.source_runs.ead}; no metrics are "
            "recalculated."
        ),
        scenario_analysis=(
            f"Base ECL is ${base_ecl / 1_000_000:,.2f}M, Upside ECL is "
            f"${upside_ecl / 1_000_000:,.2f}M, and Downside ECL is "
            f"${downside_ecl / 1_000_000:,.2f}M."
        ),
        limitations=" ".join(context.limitations),
        management_actions=(
            "Use this narrative as a review aid, not as a substitute for model owner approval, "
            "independent validation, or governance sign-off."
        ),
    )


def _prompt_payload(context: ReportContext) -> dict[str, Any]:
    return context.model_dump(exclude={"allowed_numbers", "generated_at", "request_profile"})


def _user_prompt(payload: dict[str, Any]) -> str:
    return "Use this compact validated IFRS 9 context only:\n" + json.dumps(
        payload,
        separators=(",", ":"),
        default=str,
    )


def _request_profile(
    context: ReportContext,
    user_prompt: str,
    config: ReportingConfig,
) -> dict[str, Any]:
    char_count = len(SYSTEM_PROMPT) + len(user_prompt)
    return {
        "context_char_count": char_count,
        "estimated_input_tokens": math.ceil(char_count / 4),
        "max_completion_tokens": config.max_output_tokens,
        "context_char_limit": config.context_char_limit,
        "report_context_profile": context.report_type,
    }


def _compact_prompt_payload(payload: dict[str, Any], *, list_limit: int) -> dict[str, Any]:
    compacted = json.loads(json.dumps(payload, default=str))
    compacted["limitations"] = compacted.get("limitations", [])[:5]
    for section in ("staging", "ecl", "pd", "lgd", "ead", "monitoring", "scenario"):
        if isinstance(compacted.get(section), dict):
            _limit_lists(compacted[section], list_limit)
    return compacted


def _limit_lists(value: Any, limit: int) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, list):
                value[key] = item[:limit]
                for nested in value[key]:
                    _limit_lists(nested, limit)
            else:
                _limit_lists(item, limit)
    elif isinstance(value, list):
        del value[limit:]
        for item in value:
            _limit_lists(item, limit)


def _extract_numbers(text: str) -> list[str]:
    pattern = r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?(?:[BM])?"
    return re.findall(pattern, text)


def _is_ignorable_number(value: str) -> bool:
    clean = value.strip("$%,BM").replace(",", "")
    if clean in {"1", "2", "3", "9", "12"}:
        return True
    if clean in {"2025", "03", "01"}:
        return True
    return False


def _normalized_number(value: str) -> str:
    return value.replace(",", "")


def _heading_to_html(line: str) -> str:
    if line.startswith("# "):
        return f"<h1>{line[2:]}</h1>"
    if line.startswith("## "):
        return f"<h2>{line[3:]}</h2>"
    return ""
