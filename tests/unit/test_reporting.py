from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

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
from ifrs9.reporting.generator import (
    _generate_with_groq,
    groq_messages,
    groq_request_profile,
    groq_response_format,
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
            portfolio_position=None,
            key_risk_movements=None,
            model_performance=None,
            scenario_analysis=None,
            limitations="AI-assisted commentary over validated artifacts.",
            management_actions=None,
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
        portfolio_position=None,
        key_risk_movements=None,
        model_performance=None,
        scenario_analysis=None,
        limitations=None,
        management_actions=None,
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


def test_groq_response_format_is_explicit_minimal_strict_schema() -> None:
    response_format = groq_response_format()
    schema = response_format["json_schema"]["schema"]
    expected_fields = [
        "title",
        "executive_summary",
        "portfolio_position",
        "key_risk_movements",
        "model_performance",
        "scenario_analysis",
        "limitations",
        "management_actions",
    ]

    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "ifrs9_report",
            "strict": True,
            "schema": {
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
                "required": expected_fields,
                "additionalProperties": False,
            },
        },
    }
    assert schema["additionalProperties"] is False
    assert schema["required"] == expected_fields
    assert set(schema["properties"]) == set(expected_fields)
    for field in expected_fields[1:]:
        assert schema["properties"][field]["type"] == ["string", "null"]
    assert schema["properties"]["title"]["type"] == "string"
    _assert_no_nested_schema_features(response_format)


def test_report_type_projection_excludes_unrelated_sections() -> None:
    executive = build_report_context(
        repo_root=Path.cwd(),
        report_type="Executive IFRS 9 Summary",
        audience="Risk Committee",
        detail="Standard",
    )
    lgd = build_report_context(
        repo_root=Path.cwd(),
        report_type="LGD Model Review",
        audience="Model Validation",
        detail="Standard",
    )

    assert executive.portfolio["total_ead"] > 0
    assert executive.ecl["by_stage"]
    assert executive.pd == {}
    assert executive.lgd == {}
    assert executive.monitoring == {}
    assert "scorecard_metrics" not in executive.model_dump_json()
    assert "backtesting_by_split" not in executive.lgd
    assert lgd.lgd["run_id"] == "lgd_v1_2"
    assert lgd.pd == {}
    assert "rating_summary" not in lgd.model_dump_json()


def test_all_report_contexts_are_compact_and_non_loan_level() -> None:
    config = load_reporting_config(Path.cwd())
    for report_type in REPORT_TYPES:
        context = build_report_context(
            repo_root=Path.cwd(),
            report_type=report_type,
            audience="Risk Committee",
            detail="Standard",
        )
        profile = groq_request_profile(context, config)
        payload = context.model_dump_json()

        assert "loan_id" not in payload
        assert "part-ecl.parquet" not in payload
        assert profile["context_char_count"] <= config.context_char_limit
        assert profile["max_completion_tokens"] == 1200


def test_oversized_prompt_context_is_safely_compacted() -> None:
    config = load_reporting_config(Path.cwd())
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Scorecard / PD Model Report",
        audience="Technical",
        detail="Detailed",
    )
    huge_rows = [{"metric": f"metric_{idx}", "value": idx} for idx in range(500)]
    oversized = context.model_copy(
        deep=True,
        update={"pd": {"run_id": "pd_behavioural_qe_v1", "large_optional_diagnostic": huge_rows}},
    )

    messages, profile = groq_messages(oversized, config)

    assert profile["context_char_count"] <= config.context_char_limit
    user_message = messages[1]["content"]
    assert "metric_0" in user_message
    assert "metric_3" not in user_message


def test_structured_output_schema_remains_unchanged() -> None:
    schema = groq_response_format()["json_schema"]["schema"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["executive_summary"]["type"] == ["string", "null"]
    assert schema["required"] == [
        "title",
        "executive_summary",
        "portfolio_position",
        "key_risk_movements",
        "model_performance",
        "scenario_analysis",
        "limitations",
        "management_actions",
    ]


def test_groq_request_uses_strict_compatible_schema(monkeypatch) -> None:
    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Executive IFRS 9 Summary",
        audience="Risk Committee",
        detail="Standard",
    )
    config = load_reporting_config(Path.cwd())
    captured: dict[str, Any] = {}
    payload = GeneratedReport(
        title="Executive IFRS 9 Summary",
        executive_summary="AI-assisted summary.",
        portfolio_position=None,
        key_risk_movements=None,
        model_performance=None,
        scenario_analysis=None,
        limitations=None,
        management_actions=None,
    ).model_dump_json()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content=payload)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class _FakeGroq:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=_FakeCompletions().create),
            )

    fake_module = types.ModuleType("groq")
    fake_module.Groq = _FakeGroq
    monkeypatch.setitem(sys.modules, "groq", fake_module)

    report = _generate_with_groq(context, config, "test-key")

    response_format = captured["response_format"]
    assert report.title == "Executive IFRS 9 Summary"
    assert response_format == groq_response_format()
    _assert_no_nested_schema_features(response_format)
    json.dumps(response_format)


def _assert_no_nested_schema_features(payload: Any) -> None:
    forbidden = {"$defs", "$ref", "anyOf", "allOf", "oneOf"}
    if isinstance(payload, dict):
        assert forbidden.isdisjoint(payload)
        for value in payload.values():
            _assert_no_nested_schema_features(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_nested_schema_features(item)
