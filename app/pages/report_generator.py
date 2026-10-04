"""AI-assisted report generator page."""

from __future__ import annotations

import streamlit as st
from app.components.layout import friendly_error, page_title
from app.services.reporting import (
    AUDIENCES,
    DETAIL_LEVELS,
    REPORT_TYPES,
    SCENARIO_RUNS,
    build_context_for_ui,
    export_payloads,
    generate_report_for_ui,
    groq_key_available,
)


@st.cache_data(show_spinner=False)
def _context_preview(report_type: str, audience: str, detail: str, scenario_run: str):
    return build_context_for_ui(
        report_type=report_type,
        audience=audience,
        detail=detail,
        scenario_run=scenario_run,
    )


def main() -> None:
    page_title(
        "Report Generator",
        "Groq Free Tier AI-assisted commentary over validated IFRS 9 artifacts.",
    )
    st.warning(
        "Governance boundary: the report generator cannot calculate or modify PD, LGD, EAD, "
        "SICR, Stage, ECL, scenarios, or model metrics. Numbers must originate from the "
        "structured report context."
    )

    c1, c2, c3, c4 = st.columns(4)
    report_type = c1.selectbox("Report type", REPORT_TYPES)
    scenario_run = c2.selectbox("Selected run / scenario", SCENARIO_RUNS, index=1)
    audience = c3.selectbox("Audience", AUDIENCES, index=1)
    detail = c4.selectbox("Detail", DETAIL_LEVELS, index=1)
    use_llm = st.toggle(
        "Use Groq LLM when key is available",
        value=groq_key_available(),
        disabled=not groq_key_available(),
    )

    if not groq_key_available():
        st.info(
            "LLM reporting unavailable. All quantitative application pages remain functional. "
            "A deterministic template summary is available. Configure `GROQ_API_KEY` to enable "
            "Groq generation."
        )

    try:
        context = _context_preview(report_type, audience, detail, scenario_run)
    except Exception as exc:
        friendly_error(exc)
        return

    with st.expander("Source runs and methodology limitations", expanded=True):
        st.json(context.source_runs.model_dump())
        for item in context.limitations:
            st.caption(f"- {item}")

    if st.button("Generate Report", type="primary"):
        try:
            with st.spinner("Generating report from validated context..."):
                result, context_payload = generate_report_for_ui(
                    report_type=report_type,
                    audience=audience,
                    detail=detail,
                    scenario_run=scenario_run,
                    use_llm=use_llm,
                )
        except Exception as exc:
            friendly_error(exc)
            return

        st.session_state["last_report_result"] = result.model_dump()
        st.session_state["last_report_context"] = context_payload

    if "last_report_result" not in st.session_state:
        return

    from ifrs9.reporting.generator import ReportGenerationResult

    result = ReportGenerationResult.model_validate(st.session_state["last_report_result"])
    report = result.report
    st.caption(f"Generated timestamp: `{result.generated_at}`")
    st.caption(f"Source: `{result.source}`")
    st.info("AI-assisted commentary. Review source artifacts and validation warnings before use.")
    if result.unsupported_numbers:
        st.error("Unsupported numerical claims detected.")
        st.write(result.unsupported_numbers)
    for warning in result.warnings:
        st.warning(warning)

    sections = [
        ("Executive Summary", report.executive_summary),
        ("Portfolio Position", report.portfolio_position),
        ("Key Risk Movements", report.key_risk_movements),
        ("Model Performance", report.model_performance),
        ("Scenario Analysis", report.scenario_analysis),
        ("Limitations", report.limitations),
        ("Management Actions", report.management_actions),
    ]
    st.header(report.title)
    for heading, body in sections:
        if body:
            with st.expander(heading, expanded=heading == "Executive Summary"):
                st.write(body)

    payloads = export_payloads(result)
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download Markdown",
        payloads["markdown"],
        file_name="ifrs9_report.md",
        mime="text/markdown",
    )
    d2.download_button(
        "Download JSON",
        payloads["json"],
        file_name="ifrs9_report.json",
        mime="application/json",
    )
    d3.download_button(
        "Download HTML",
        payloads["html"],
        file_name="ifrs9_report.html",
        mime="text/html",
    )


main()
