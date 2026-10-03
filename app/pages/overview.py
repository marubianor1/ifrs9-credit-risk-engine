"""Overview page for persisted ECL results."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, money, page_title, pct
from app.services.ecl import ecl_kpis, load_ecl_artifacts


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_ecl_artifacts(run_id)


def main() -> None:
    page_title("Overview", "Portfolio ECL view from persisted `ecl_v1` artifacts.")
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

    cols = st.columns(3)
    cols[0].metric("Total EAD", money(kpis["total_ead"]))
    cols[1].metric("Weighted ECL", money(kpis["weighted_ecl"]))
    cols[2].metric("Coverage ratio", pct(kpis["coverage_ratio"]))

    left, right = st.columns(2)
    with left:
        st.subheader("EAD and ECL by Stage")
        stage_chart = stage.melt(
            id_vars=["stage"],
            value_vars=["total_ead", "weighted_ecl"],
            var_name="metric",
            value_name="amount",
        )
        fig = px.bar(stage_chart, x="stage", y="amount", color="metric", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stage, use_container_width=True, hide_index=True)

    with right:
        st.subheader("ECL by Scenario")
        fig = px.bar(scenario, x="scenario", y="ecl", color="scenario")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(scenario, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Rating Distribution")
        fig = px.bar(rating, x="rating", y="total_ead", color="rating")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("ECL by Rating")
        fig = px.bar(rating, x="rating", y="weighted_ecl", color="rating")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Methodological Limitations")
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
        with st.expander("Alignment and data-quality notes"):
            st.dataframe(warnings, use_container_width=True, hide_index=True)


main()
