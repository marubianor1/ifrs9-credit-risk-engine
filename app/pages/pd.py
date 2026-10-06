"""PD framework page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.services.pd import available_pd_runs, load_pd_artifacts, run_pd_from_controls
from app.services.runtime import full_mode_message, is_cloud_demo


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_pd_artifacts(run_id)


def main() -> None:
    page_title("PD", "Calibration, rating scale, lifetime PD and backtesting artifacts.")
    render_guidance("pd")
    runs = available_pd_runs()
    if not runs:
        st.warning("No PD artifacts are available.")
        return
    selected = st.selectbox(
        "PD run",
        runs,
        index=runs.index(st.session_state.get("pd_run", runs[0]))
        if st.session_state.get("pd_run", runs[0]) in runs
        else 0,
    )
    st.session_state["pd_run"] = selected

    try:
        artifacts = _load(selected)
    except Exception as exc:
        friendly_error(exc)
        return

    st.subheader("Future Run Controls")
    if is_cloud_demo():
        st.info(full_mode_message())
    else:
        c1, c2, c3, c4 = st.columns(4)
        calibration_method = c1.selectbox(
            "Calibration method",
            ["split_mean_anchor", "isotonic", "platt"],
        )
        anchor_dates = c2.text_input("Anchor dates", "2012-01-01 to 2024-03-01")
        rating_config = c3.text_input("Rating configuration", "5-grade master scale")
        horizon = c4.number_input(
            "Max lifetime horizon",
            min_value=12,
            max_value=600,
            value=360,
            step=12,
        )
        if st.button("Run PD Framework", type="primary"):
            st.caption(
                "Calibration method, anchor dates, and rating configuration are captured for "
                "future UI overrides; the current PD backend uses the versioned YAML "
                "configuration."
            )
            _ = (calibration_method, anchor_dates, rating_config)
            with st.spinner("Running PD framework via existing backend API..."):
                try:
                    new_run_id = run_pd_from_controls(
                        scorecard_run_id=st.session_state.get("scorecard_run", "behavioural_qe_v1"),
                        max_lifetime_horizon=int(horizon),
                    )
                except Exception as exc:
                    friendly_error(exc)
                else:
                    st.session_state["pd_run"] = new_run_id
                    st.success(f"Created PD run `{new_run_id}`")

    tab_cal, tab_rating, tab_lifetime, tab_backtest = st.tabs(
        ["Calibration", "Ratings", "Lifetime PD", "Backtesting"]
    )
    with tab_cal:
        st.dataframe(artifacts["calibration_metrics"], use_container_width=True, hide_index=True)
        curve = artifacts["calibration_curve"]
        if {"predicted_bad_rate", "observed_bad_rate"}.issubset(curve.columns):
            fig = px.line(
                curve,
                x="predicted_bad_rate",
                y="observed_bad_rate",
                color="split",
                markers=True,
                title="Raw vs Calibrated PD Calibration",
            )
            st.plotly_chart(fig, use_container_width=True)
    with tab_rating:
        left, right = st.columns(2)
        with left:
            st.subheader("Rating Master Scale")
            st.dataframe(artifacts["rating_summary"], use_container_width=True, hide_index=True)
            st.dataframe(artifacts["rating_boundaries"], use_container_width=True, hide_index=True)
        with right:
            st.subheader("TTC PD")
            st.dataframe(artifacts["rating_ttc"], use_container_width=True, hide_index=True)
            fig = px.bar(artifacts["rating_ttc"], x="rating", y="rating_ttc_pd", color="rating")
            st.plotly_chart(fig, use_container_width=True)
    with tab_lifetime:
        lifetime = artifacts["lifetime"]
        available_ratings = sorted(lifetime["rating"].unique())
        ratings = st.multiselect(
            "Ratings",
            available_ratings,
            default=available_ratings[:3],
        )
        filtered = lifetime[lifetime["rating"].isin(ratings)]
        fig = px.line(filtered, x="month", y="cumulative_pd", color="rating")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("PIT / TTC Diagnostics")
        st.dataframe(artifacts["pit_ttc"], use_container_width=True, hide_index=True)
    with tab_backtest:
        st.subheader("Backtesting by Split")
        st.dataframe(artifacts["backtesting_split"], use_container_width=True, hide_index=True)
        st.subheader("Yearly Rating Backtesting")
        yearly = artifacts["backtesting_year"]
        fig = px.line(yearly, x="observation_year", y="oe_ratio", color="rating", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Transition Matrix")
        st.dataframe(artifacts["transition"], use_container_width=True, hide_index=True)
        st.dataframe(artifacts["transition_summary"], use_container_width=True, hide_index=True)


main()
