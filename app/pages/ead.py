"""EAD framework page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, page_title
from app.services.ead import (
    BASELINE_METHOD,
    available_ead_runs,
    load_ead_artifacts,
    run_ead_from_controls,
)
from app.services.runtime import full_mode_message, is_cloud_demo


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_ead_artifacts(run_id)


def main() -> None:
    page_title("EAD", "Exposure at default diagnostics for amortizing mortgages.")
    runs = available_ead_runs()
    selected = st.selectbox("EAD run", runs, index=runs.index("ead_v1") if "ead_v1" in runs else 0)
    st.info(f"Selected baseline method = `{BASELINE_METHOD}`")
    try:
        artifacts = _load(selected)
    except Exception as exc:
        friendly_error(exc)
        return

    ratio = artifacts["ratio_distribution"]
    split = artifacts["backtest_split"]
    segment = artifacts["backtest_segment"]
    profiles = artifacts["profiles"]

    st.subheader("Future EAD Run Controls")
    if is_cloud_demo():
        st.info(full_mode_message())
    else:
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox(
            "Method",
            ["contractual_amortization", "current_balance", "empirical_model"],
        )
        max_horizon = c2.number_input(
            "Max horizon",
            min_value=12,
            max_value=600,
            value=360,
            step=12,
        )
        amortization = c3.selectbox("Amortization assumption", ["standard", "zero_interest_safe"])
        if st.button("Run EAD", type="primary"):
            _ = (method, max_horizon, amortization)
            with st.spinner("Running EAD framework via existing backend API..."):
                try:
                    new_run = run_ead_from_controls()
                except Exception as exc:
                    friendly_error(exc)
                else:
                    st.success(f"Created EAD run `{new_run}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Observable defaults", f"{ratio['observable_defaults'].iloc[0]:,.0f}")
    c2.metric("Mean EAD ratio", f"{ratio['ead_ratio_mean'].iloc[0]:.3f}")
    c3.metric("Median EAD ratio", f"{ratio['ead_ratio_median'].iloc[0]:.3f}")

    tab_backtest, tab_segments, tab_profiles = st.tabs(["Backtesting", "Segments", "Profiles"])
    with tab_backtest:
        st.subheader("Current Balance vs Contractual vs Empirical")
        split_fig = px.bar(split, x="split", y="oe_ratio", color="method", barmode="group")
        st.plotly_chart(split_fig, use_container_width=True)
        st.dataframe(split, use_container_width=True, hide_index=True)
    with tab_segments:
        mtd = st.selectbox("Method for segment view", sorted(segment["method"].unique()))
        filtered = segment[segment["method"].eq(mtd)]
        segment_fig = px.bar(filtered, x="segment_value", y="oe_ratio", color="split")
        st.plotly_chart(segment_fig, use_container_width=True)
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    with tab_profiles:
        sample = profiles.head(2000)
        value_cols = [col for col in sample.columns if col.startswith("ead_")]
        if value_cols:
            profile = sample.groupby("month", observed=True)[value_cols].mean().reset_index()
            fig = px.line(profile, x="month", y=value_cols)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(artifacts["profile_validation"], use_container_width=True, hide_index=True)


main()
