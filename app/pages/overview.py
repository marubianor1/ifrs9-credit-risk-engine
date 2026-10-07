"""V3 Risk Committee overview page for persisted ECL results."""

from __future__ import annotations

import streamlit as st
from app.components.advisory import (
    apply_advisory_css,
    context_bar,
    downside_observation,
    executive_kpis,
    insight_callout,
    methodology_note,
    page_header,
    rating_concentration_insight,
    section_header,
    source_caption,
    stage_concentration_insight,
    stage_concentration_table,
    stage_coverage_observation,
)
from app.components.charts import (
    horizontal_contribution_bar,
    scenario_delta_bar,
    stage_concentration_chart,
)
from app.components.formatting import au_date, percentage, signed_usd, usd
from app.components.guidance import render_guidance
from app.components.layout import friendly_error
from app.components.tables import format_table
from app.services.ecl import ecl_kpis, load_ecl_artifacts


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_ecl_artifacts(run_id)


def main() -> None:
    apply_advisory_css()
    page_header(
        "IFRS 9 Portfolio Risk",
        "Risk Committee summary of exposure, staging and expected credit loss concentration.",
    )
    try:
        artifacts = _load(st.session_state.get("ecl_run", "ecl_v1"))
    except Exception as exc:
        friendly_error(exc)
        return

    stage = artifacts["stage"]
    rating = artifacts["rating"]
    scenario = artifacts["scenario"]
    warnings = artifacts["warnings"]
    kpis = ecl_kpis(stage, scenario)
    stage_23_ead = float(stage.loc[stage["stage"].isin([2, 3]), "total_ead"].sum())
    stage_23_share = stage_23_ead / kpis["total_ead"] if kpis["total_ead"] else 0.0
    base_ecl = float(scenario.loc[scenario["scenario"].eq("BASE"), "ecl"].iloc[0])
    downside_ecl = float(scenario.loc[scenario["scenario"].eq("DOWNSIDE"), "ecl"].iloc[0])
    downside_impact = downside_ecl - base_ecl

    context_bar(
        [
            ("As at", au_date("2025-03-01")),
            ("Run", "ecl_v1"),
            ("Portfolio", "Freddie Mac Case Study"),
            ("Currency", "USD"),
        ]
    )
    insight = stage_concentration_insight(stage, source="As at 1 Mar 2025 · Source: ecl_v1")
    if insight:
        insight_callout(insight)

    executive_kpis(
        [
            ("Portfolio EAD", usd(kpis["total_ead"]), "Current reporting-date exposure"),
            ("Weighted ECL", usd(kpis["weighted_ecl"]), "Probability-weighted IFRS 9 ECL"),
            ("Coverage ratio", percentage(kpis["coverage_ratio"]), "Weighted ECL / EAD"),
        ],
        columns=3,
    )
    executive_kpis(
        [
            ("Stage 2 + 3 exposure share", percentage(stage_23_share), "Higher-risk stage EAD"),
            ("Downside impact", signed_usd(downside_impact), "Downside ECL less Base"),
        ],
        columns=2,
    )

    section_header(
        "Stage concentration",
        "Compares portfolio exposure share with expected credit loss share by IFRS 9 stage.",
    )
    st.plotly_chart(
        stage_concentration_chart(stage, title="Exposure share versus ECL share by stage"),
        width="stretch",
    )
    source_caption("As at 1 Mar 2025 · Source: ecl_v1")

    left, right = st.columns([1, 1])
    with left:
        section_header("Scenario sensitivity", "ECL movement relative to Base scenario.")
        st.plotly_chart(
            scenario_delta_bar(scenario, title="ECL delta versus Base"),
            width="stretch",
        )
        source_caption("As at 1 Mar 2025 · Source: ecl_v1")
    with right:
        section_header("Rating contribution", "Ranked ECL contribution by portfolio rating.")
        ranked = rating.sort_values("weighted_ecl", ascending=False).copy()
        st.plotly_chart(
            horizontal_contribution_bar(
                ranked,
                label="rating",
                value="weighted_ecl",
                title="Weighted ECL by rating",
                xaxis_title="Weighted ECL",
            ),
            width="stretch",
        )
        source_caption("As at 1 Mar 2025 · Source: ecl_v1")

    section_header(
        "Key observations",
        "Deterministic observations from validated aggregate outputs.",
    )
    observations = [
        stage_coverage_observation(stage),
        downside_observation(scenario),
    ]
    rating_insight = rating_concentration_insight(
        rating,
        source="As at 1 Mar 2025 · Source: ecl_v1",
    )
    if rating_insight:
        observations.append(rating_insight.text)
    for observation in [item for item in observations if item]:
        st.markdown(f"- {observation}")

    section_header("Stage analytical table", "EAD, ECL and concentration metrics by stage.")
    st.dataframe(format_table(stage_concentration_table(stage)), width="stretch", hide_index=True)

    methodology_note(
        "LGD is structural and macro-neutral in the baseline. Stage 3 uses the documented "
        "current-exposure times LGD approximation where detailed default cashflow timing is "
        "unavailable. Scenario weights and downturn LGD are sensitivities, not refitted models."
    )
    render_guidance("overview")
    if not warnings.empty:
        with st.expander("Technical diagnostics", expanded=False):
            st.dataframe(format_table(warnings), width="stretch", hide_index=True)


main()
