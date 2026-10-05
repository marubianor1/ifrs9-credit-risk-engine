"""AI-assisted report generation with deterministic fallback."""

from __future__ import annotations

import json
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


def generate_report(
    *,
    context: ReportContext,
    config: ReportingConfig,
    api_key: str | None = None,
    use_llm: bool = True,
) -> ReportGenerationResult:
    """Generate a structured report from validated context only."""
    if not use_llm:
        return _fallback_result(context, "LLM disabled by user selection.")
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        return _fallback_result(context, "GROQ_API_KEY is not configured.")
    if config.provider.lower() != "groq":
        return _fallback_result(context, f"Unsupported reporting provider: {config.provider}.")
    try:
        report = _generate_with_groq(context, config, resolved_key)
    except Exception as exc:
        return _fallback_result(context, f"Groq generation failed: {exc}")

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
    prompt = _prompt(context)
    schema = groq_strict_json_schema(GeneratedReport)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise IFRS 9 portfolio commentary using only supplied JSON. "
                    "Never calculate, infer, or invent metrics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": config.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ifrs9_report",
                "schema": schema,
                "strict": True,
            },
        },
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


def groq_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a Groq strict Structured Outputs compatible JSON Schema."""
    schema = model.model_json_schema()
    _close_object_schemas(schema)
    return schema


def _close_object_schemas(schema: Any) -> None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        schema_type = schema.get("type")
        if schema_type == "object" or isinstance(properties, dict):
            schema["additionalProperties"] = False
            schema["required"] = list(properties) if isinstance(properties, dict) else []
        for key in ("properties", "$defs"):
            nested = schema.get(key)
            if isinstance(nested, dict):
                for item in nested.values():
                    _close_object_schemas(item)
        for key in ("items", "anyOf", "allOf", "oneOf"):
            nested = schema.get(key)
            _close_object_schemas(nested)
    elif isinstance(schema, list):
        for item in schema:
            _close_object_schemas(item)


def _fallback_result(context: ReportContext, reason: str) -> ReportGenerationResult:
    report = _fallback_report(context)
    return ReportGenerationResult(
        report=report,
        generated_at=datetime.now(UTC).isoformat(),
        source="deterministic_template",
        source_runs=context.source_runs.model_dump(),
        warnings=[reason, "LLM reporting unavailable. Quantitative pages remain functional."],
    )


def _fallback_report(context: ReportContext) -> GeneratedReport:
    portfolio = context.portfolio
    ecl = context.ecl
    scenario = context.scenario
    return GeneratedReport(
        title=f"{context.report_type} - {context.reporting_date}",
        executive_summary=(
            f"Portfolio weighted ECL is ${portfolio['weighted_ecl'] / 1_000_000:,.2f}M "
            f"on total EAD of ${portfolio['total_ead'] / 1_000_000_000:,.2f}B, with "
            f"coverage of {portfolio['coverage_ratio'] * 100:.2f}%."
        ),
        portfolio_position=(
            f"Stage 1, Stage 2, and Stage 3 results are sourced from "
            f"{context.source_runs.ecl} and {context.source_runs.staging}."
        ),
        key_risk_movements=(
            f"The selected scenario run {scenario['run_id']} changes ECL by "
            f"${float(scenario['delta_ecl']) / 1_000_000:,.2f}M versus baseline."
        ),
        model_performance=(
            f"PD, LGD, and EAD diagnostics are sourced from {context.source_runs.pd}, "
            f"{context.source_runs.lgd}, and {context.source_runs.ead}; no metrics are "
            "recalculated."
        ),
        scenario_analysis=(
            f"Base ECL is ${ecl['base_ecl'] / 1_000_000:,.2f}M, Upside ECL is "
            f"${ecl['upside_ecl'] / 1_000_000:,.2f}M, and Downside ECL is "
            f"${ecl['downside_ecl'] / 1_000_000:,.2f}M."
        ),
        limitations=" ".join(context.limitations),
        management_actions=(
            "Use this narrative as a review aid, not as a substitute for model owner approval, "
            "independent validation, or governance sign-off."
        ),
    )


def _prompt(context: ReportContext) -> str:
    return (
        "You are drafting AI-assisted IFRS 9 portfolio commentary. "
        "Use only the JSON context below. Do not calculate or invent metrics. "
        "If a number is not present in the context, omit it. "
        "Return only JSON matching the provided schema.\n\n"
        f"{context.model_dump_json(indent=2)}"
    )


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
