"""ECL engine page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, money, page_title, pct
from app.services.ecl import ecl_kpis, load_ecl_artifacts, run_ecl_from_ui


@st.cache_data(show_spinner=False)
def _load():
    return load_ecl_artifacts("ecl_v1")


def main() -> None:
    page_title("ECL", "Expected credit loss engine outputs and sensitivities.")
    try:
        artifacts = _load()
    except Exception as exc:
        friendly_error(exc)
        return
    stage = artifacts["stage"]
    rating = artifacts["rating"]
    scenario = artifacts["scenario"]
    kpis = ecl_kpis(stage, scenario)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total EAD", money(kpis["total_ead"]))
    c2.metric("Weighted ECL", money(kpis["weighted_ecl"]))
    c3.metric("Coverage ratio", pct(kpis["coverage_ratio"]))
    s1, s2, s3 = st.columns(3)
    for column, stage_id in zip([s1, s2, s3], [1, 2, 3], strict=True):
        value = stage.loc[stage["stage"].eq(stage_id), "weighted_ecl"].iloc[0]
        column.metric(f"Stage {stage_id} ECL", money(value))

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
        st.subheader("EAD / ECL by Stage")
        stage_chart = stage.melt(
            id_vars=["stage"],
            value_vars=["total_ead", "weighted_ecl"],
            var_name="metric",
            value_name="amount",
        )
        amount_fig = px.bar(stage_chart, x="stage", y="amount", color="metric")
        st.plotly_chart(amount_fig, use_container_width=True)
        st.subheader("Coverage by Stage")
        coverage_fig = px.bar(stage, x="stage", y="coverage_ratio", color="stage")
        st.plotly_chart(coverage_fig, use_container_width=True)
    with right:
        st.subheader("Base / Upside / Downside")
        scenario_fig = px.bar(scenario, x="scenario", y="ecl", color="scenario")
        st.plotly_chart(scenario_fig, use_container_width=True)
        st.subheader("ECL by Rating")
        rating_fig = px.bar(rating, x="rating", y="weighted_ecl", color="rating")
        st.plotly_chart(rating_fig, use_container_width=True)

    st.subheader("Stage 1 12M vs Stage 2 Lifetime")
    st.dataframe(stage, use_container_width=True, hide_index=True)
    st.subheader("Structural LGD vs Downturn Sensitivity")
    st.dataframe(artifacts["downturn"], use_container_width=True, hide_index=True)
    st.subheader("Alignment Notes")
    st.dataframe(artifacts["warnings"], use_container_width=True, hide_index=True)

    if st.button("Run ECL", type="primary"):
        try:
            with st.spinner("Running ECL backend..."):
                result = run_ecl_from_ui()
        except Exception as exc:
            friendly_error(exc)
        else:
            st.success(f"ECL run completed: `{result.run_id}`")


main()
