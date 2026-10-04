from __future__ import annotations

from pathlib import Path

from ifrs9.reporting import (
    REPORT_TYPES,
    GeneratedReport,
    build_report_context,
    generate_report,
    load_reporting_config,
    render_report_html,
    render_report_markdown,
    validate_report_numbers,
)


def test_report_context_generation_has_no_loan_level_payload() -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type=REPORT_TYPES[0],
        audience="Risk Committee",
        detail="Standard",
    )
    payload = context.model_dump_json()

    assert context.portfolio["total_ead"] > 0
    assert context.source_runs.ecl == "ecl_v1"
    assert "loan_id" not in payload
    assert "part-ecl.parquet" not in payload
    assert "ead_observations.parquet" not in payload


def test_missing_key_fallback(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Provisioning Report",
        audience="Executive",
        detail="Concise",
    )
    config = load_reporting_config(Path.cwd())

    result = generate_report(context=context, config=config, api_key=None, use_llm=True)

    assert result.source == "deterministic_template"
    assert "GROQ_API_KEY" in result.warnings[0]


def test_rate_limit_or_provider_error_fallback(monkeypatch) -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Scenario Stress Report",
        audience="Technical",
        detail="Detailed",
    )
    config = load_reporting_config(Path.cwd())

    def fail_provider(*args, **kwargs):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr("ifrs9.reporting.generator._generate_with_groq", fail_provider)
    result = generate_report(context=context, config=config, api_key="test-key", use_llm=True)

    assert result.source == "deterministic_template"
    assert "429 rate limit" in result.warnings[0]


def test_structured_output_parsing_with_mocked_groq(monkeypatch) -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Executive IFRS 9 Summary",
        audience="Risk Committee",
        detail="Standard",
    )
    config = load_reporting_config(Path.cwd())

    def fake_provider(*args, **kwargs):
        return GeneratedReport(
            title="Executive IFRS 9 Summary",
            executive_summary="Portfolio weighted ECL is $690.09M.",
            limitations="AI-assisted commentary over validated artifacts.",
        )

    monkeypatch.setattr("ifrs9.reporting.generator._generate_with_groq", fake_provider)
    result = generate_report(context=context, config=config, api_key="test-key", use_llm=True)

    assert result.source == "groq_structured_output"
    assert result.report.executive_summary
    assert result.unsupported_numbers == []


def test_unsupported_number_validation() -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Model Monitoring Report",
        audience="Model Validation",
        detail="Standard",
    )
    report = GeneratedReport(
        title="Bad Report",
        executive_summary="The portfolio has an unsupported loss estimate of $123.45M.",
    )

    unsupported = validate_report_numbers(report, context)

    assert "$123.45M" in unsupported


def test_export_generation() -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="EAD Model Review",
        audience="Technical",
        detail="Concise",
    )
    config = load_reporting_config(Path.cwd())
    result = generate_report(context=context, config=config, use_llm=False)

    markdown = render_report_markdown(result)
    html = render_report_html(result)

    assert markdown.startswith("# ")
    assert "<html>" in html
    assert result.model_dump()["report"]["title"]
