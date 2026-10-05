"""ECL engine page."""

from __future__ import annotations

import streamlit as st
from app.components.charts import rating_bar, scenario_bar, stage_bar
from app.components.formatting import au_date, percentage, usd
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
from app.services.ecl import ecl_kpis, load_ecl_artifacts, run_ecl_from_ui
from app.services.runtime import full_mode_message, is_cloud_demo


@st.cache_data(show_spinner=False)
def _load():
    return load_ecl_artifacts("ecl_v1")


def main() -> None:
    page_title(
        "ECL",
        "Expected credit loss engine outputs, scenario comparison and LGD sensitivity.",
    )
    try:
        artifacts = _load()
    except Exception as exc:
        friendly_error(exc)
        return
    stage = artifacts["stage"]
    rating = artifacts["rating"]
    scenario = artifacts["scenario"]
    kpis = ecl_kpis(stage, scenario)
    stage_ecl = {
        int(row["stage"]): float(row["weighted_ecl"])
        for _, row in stage.iterrows()
    }

    st.caption("Selected run: `ecl_v1` | Reporting date: " + au_date("2025-03-01"))
    kpi_row(
        [
            ("Total EAD", usd(kpis["total_ead"]), "Exposure used for ECL."),
            ("Weighted ECL", usd(kpis["weighted_ecl"]), "Probability-weighted ECL."),
            ("Coverage ratio", percentage(kpis["coverage_ratio"]), "Weighted ECL / EAD."),
            ("Stage 1 ECL", usd(stage_ecl[1]), "12-month ECL."),
            ("Stage 2 ECL", usd(stage_ecl[2]), "Lifetime ECL for SICR loans."),
            ("Stage 3 ECL", usd(stage_ecl[3]), "Credit-impaired approximation."),
        ]
    )

    with st.expander("Methodological limitations"):
        st.markdown(
            """
            - Baseline LGD is structural and macro-neutral.
            - Stage 3 is approximated as current exposure times LGD.
            - EAD uses contractual amortization as the selected baseline.
            - Quarterly macro anchors label PD scenario paths; monthly ECL horizons start at
              month 1 after the reporting date.
            """
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            stage_bar(
                stage,
                value="weighted_ecl",
                title="Weighted ECL by IFRS 9 stage",
                yaxis_title="Weighted ECL",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            stage_bar(
                stage,
                value="coverage_ratio",
                title="Coverage ratio by IFRS 9 stage",
                yaxis_title="Coverage ratio",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            scenario_bar(scenario, title="Scenario ECL under Base, Upside and Downside"),
            use_container_width=True,
        )
        st.plotly_chart(
            rating_bar(
                rating,
                value="weighted_ecl",
                title="Rating contribution to weighted ECL",
                yaxis_title="Weighted ECL",
            ),
            use_container_width=True,
        )

    with st.expander("Stage 1 12M vs Stage 2 lifetime reconciliation", expanded=False):
        st.dataframe(format_table(stage), use_container_width=True, hide_index=True)
    with st.expander("Structural LGD versus downturn sensitivity", expanded=True):
        st.dataframe(format_table(artifacts["downturn"]), use_container_width=True, hide_index=True)
    with st.expander("Alignment and data-quality notes", expanded=False):
        st.dataframe(format_table(artifacts["warnings"]), use_container_width=True, hide_index=True)

    if is_cloud_demo():
        st.info(full_mode_message())
    elif st.button("Run ECL", type="primary"):
        try:
            with st.spinner("Running ECL backend..."):
                result = run_ecl_from_ui()
        except Exception as exc:
            friendly_error(exc)
        else:
            st.success(f"ECL run completed: `{result.run_id}`")


main()
