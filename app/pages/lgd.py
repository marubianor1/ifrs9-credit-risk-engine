"""LGD framework page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, page_title, pct
from app.services.lgd import (
    BASELINE_RUN,
    available_lgd_runs,
    lgd_governance_status,
    lgd_population_summary,
    load_lgd_artifacts,
    load_lgd_forward_artifacts,
)


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_lgd_artifacts(run_id)


@st.cache_data(show_spinner=False)
def _load_forward(run_id: str):
    return load_lgd_forward_artifacts(run_id)


def main() -> None:
    page_title("LGD", "Loss given default diagnostics and governance.")
    governance = lgd_governance_status()
    with st.expander("Model governance status", expanded=True):
        st.write(f"Production/project baseline: `{governance['production_baseline']}`")
        st.write(f"Rejected challengers: `{', '.join(governance['rejected_challengers'])}`")
        st.caption(str(governance["reason"]))

    runs = available_lgd_runs()
    selected = st.selectbox(
        "LGD run",
        runs,
        index=runs.index(BASELINE_RUN) if BASELINE_RUN in runs else 0,
    )
    try:
        artifacts = _load(selected)
    except Exception as exc:
        friendly_error(exc)
        return

    backtesting = artifacts["backtesting"]
    components = artifacts["components"]
    by_rating = artifacts["by_rating"]
    by_year = artifacts["by_year"]
    downturn = artifacts["downturn"]
    recovery = artifacts["recovery_timing"]
    resolution = artifacts["resolution"]
    population = lgd_population_summary(artifacts["episodes"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Default episodes", f"{population['episodes']:,.0f}")
    c2.metric("Resolved episodes", f"{population['resolved']:,.0f}")
    c3.metric("Cure rate", pct(population["cure_rate"]))

    tab_perf, tab_components, tab_segments, tab_challengers = st.tabs(
        ["Performance", "Decomposition", "Segments", "Challengers"]
    )
    with tab_perf:
        st.subheader("Realized vs Predicted LGD")
        fig = px.bar(
            backtesting,
            x="split",
            y=["realized_mean_lgd", "predicted_lgd"],
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(backtesting, use_container_width=True, hide_index=True)
        st.subheader("O/E by Split")
        oe_fig = px.bar(backtesting, x="split", y="oe_ratio", color="split")
        st.plotly_chart(oe_fig, use_container_width=True)
    with tab_components:
        st.subheader("Cure vs Non-cure Decomposition")
        st.dataframe(components, use_container_width=True, hide_index=True)
        fig = px.bar(
            components,
            x="split",
            y=["observed_p_cure", "predicted_p_cure"],
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab_segments:
        left, right = st.columns(2)
        with left:
            st.subheader("LGD by Rating")
            rating_fig = px.bar(
                by_rating,
                x="rating_at_default",
                y="oe_ratio",
                color="split",
            )
            st.plotly_chart(rating_fig, use_container_width=True)
            st.dataframe(by_rating, use_container_width=True, hide_index=True)
        with right:
            st.subheader("LGD by Default Year")
            year_fig = px.line(by_year, x="default_year", y="oe_ratio", color="split")
            st.plotly_chart(year_fig, use_container_width=True)
            st.dataframe(by_year, use_container_width=True, hide_index=True)
        st.subheader("Downturn Factors")
        st.dataframe(downturn, use_container_width=True, hide_index=True)
        st.subheader("Recovery Timing")
        st.dataframe(recovery, use_container_width=True, hide_index=True)
        st.subheader("Resolution Population")
        st.dataframe(resolution, use_container_width=True, hide_index=True)
    with tab_challengers:
        st.info("LGD challenger runs are selectable for review only. The UI does not refit LGD.")
        st.dataframe(artifacts["model_metrics"], use_container_width=True, hide_index=True)
        fl_run = st.selectbox(
            "Forward-looking LGD challenger",
            ["lgd_fl_v1", "lgd_fl_v2"],
        )
        try:
            fl_artifacts = _load_forward(fl_run)
        except Exception as exc:
            friendly_error(exc)
        else:
            st.subheader("Probability-weighted LGD")
            st.dataframe(fl_artifacts["weighted"], use_container_width=True, hide_index=True)
            st.subheader("Scenario LGD by Rating")
            st.dataframe(
                fl_artifacts["scenario_by_rating"],
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Macro Relationship Diagnostics")
            st.dataframe(fl_artifacts["diagnostics"], use_container_width=True, hide_index=True)
            st.subheader("Scenario Sensitivity")
            st.dataframe(fl_artifacts["sensitivity"], use_container_width=True, hide_index=True)


main()
