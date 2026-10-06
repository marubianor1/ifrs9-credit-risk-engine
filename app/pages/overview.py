"""Overview page for persisted ECL results."""

from __future__ import annotations

import streamlit as st
from app.components.charts import rating_bar, scenario_delta_bar, stage_bar, stage_mix_bar
from app.components.formatting import au_date, percentage, signed_usd, usd
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
from app.services.ecl import ecl_kpis, load_ecl_artifacts


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_ecl_artifacts(run_id)


def main() -> None:
    page_title(
        "Overview",
        "Risk Committee view of portfolio exposure, IFRS 9 stage mix and expected credit loss.",
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

    st.caption("Selected run: `ecl_v1` | Reporting date: " + au_date("2025-03-01"))
    render_guidance("overview")
    kpi_row(
        [
            ("Portfolio EAD", usd(kpis["total_ead"]), "Current exposure at reporting date."),
            ("Weighted ECL", usd(kpis["weighted_ecl"]), "Probability-weighted ECL."),
            ("Coverage ratio", percentage(kpis["coverage_ratio"]), "Weighted ECL / EAD."),
            ("Stage 2 + 3 share", percentage(stage_23_share), "EAD in higher-risk stages."),
            ("Downside impact", signed_usd(downside_impact), "Downside ECL less Base ECL."),
        ]
    )

    stage_left, stage_mid = st.columns([1.2, 1])
    with stage_left:
        st.plotly_chart(
            stage_mix_bar(stage, title="Stage 1, Stage 2 and Stage 3 exposure mix"),
            use_container_width=True,
        )
    with stage_mid:
        st.plotly_chart(
            stage_bar(
                stage,
                value="weighted_ecl",
                title="Weighted ECL by IFRS 9 stage",
                yaxis_title="Weighted ECL",
            ),
            use_container_width=True,
        )
    stage_right, scenario_right = st.columns([1, 1])
    with stage_right:
        st.plotly_chart(
            stage_bar(
                stage,
                value="coverage_ratio",
                title="Coverage ratio by stage",
                yaxis_title="Coverage ratio",
            ),
            use_container_width=True,
        )
    with scenario_right:
        st.plotly_chart(
            scenario_delta_bar(scenario, title="Scenario sensitivity: ECL delta versus Base"),
            use_container_width=True,
        )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            rating_bar(
                rating,
                value="total_ead",
                title="Rating exposure distribution",
                yaxis_title="Exposure",
            ),
            use_container_width=True,
        )
    with right:
        st.dataframe(format_table(rating), use_container_width=True, hide_index=True)

    with st.expander("Detailed stage table", expanded=False):
        st.dataframe(format_table(stage), use_container_width=True, hide_index=True)
    with st.expander("Methodological limitations and data-quality notes", expanded=False):
        st.markdown(
            """
            - LGD is structural and macro-neutral in the baseline because macro LGD challengers
              were rejected.
            - Stage 3 uses a current-exposure times LGD approximation where detailed default
              cashflow timing is unavailable.
            - Scenario weights and downturn LGD are exposed as sensitivity controls, not
              refitted models.
            - Freddie Mac public data supports amortizing mortgage EAD, not revolving CCF modelling.
            """
        )
        if not warnings.empty:
            st.dataframe(format_table(warnings), use_container_width=True, hide_index=True)


main()
