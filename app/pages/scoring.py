"""Scoring model page for existing scorecard runs."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, page_title
from app.services.runtime import full_mode_message, is_cloud_demo
from app.services.scorecard import (
    available_scorecard_runs,
    load_scorecard_artifacts,
    run_scorecard_from_controls,
)


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_scorecard_artifacts(run_id)


def main() -> None:
    page_title("Scoring Models", "Application and behavioural scorecard diagnostics.")
    runs = available_scorecard_runs()
    if not runs:
        st.warning("No scorecard artifacts are available.")
        return
    selected = st.selectbox(
        "Scorecard run",
        runs,
        index=runs.index(st.session_state.get("scorecard_run", runs[0]))
        if st.session_state.get("scorecard_run", runs[0]) in runs
        else 0,
    )
    st.session_state["scorecard_run"] = selected

    try:
        artifacts = _load(selected)
    except Exception as exc:
        friendly_error(exc)
        return

    features = artifacts["features"]
    iv = artifacts["iv"]
    coefficients = artifacts["coefficients"]
    metrics = artifacts["metrics"]
    calibration = artifacts["calibration"]
    psi = artifacts["psi"]
    yearly = artifacts["yearly"]
    bands = artifacts["bands"]
    deciles = artifacts["deciles"]

    st.subheader("Future Run Controls")
    if is_cloud_demo():
        st.info(full_mode_message())
    else:
        c1, c2, c3 = st.columns(3)
        population = c1.selectbox("Population", ["behavioural", "application"])
        snapshot_frequency = c2.selectbox(
            "Snapshot frequency",
            ["quarter_end", "monthly", "year_end"],
        )
        sampling_strategy = c3.selectbox(
            "Sampling strategy",
            ["none", "random_nondefault", "stratified_nondefault"],
        )
        d1, d2, d3 = st.columns(3)
        train_dates = d1.text_input("Train dates", "2012-01-01 to 2018-12-31")
        validation_dates = d2.text_input("Validation dates", "2019-01-01 to 2021-12-31")
        oot_dates = d3.text_input("OOT dates", "2022-01-01 to 2024-03-01")
        if st.button("Run Model", type="primary"):
            st.caption(
                "Date windows are captured for future configuration; the current scorecard "
                "backend uses the configured development sample windows."
            )
            _ = (train_dates, validation_dates, oot_dates)
            with st.spinner("Training scorecard via existing backend API..."):
                try:
                    new_run_id = run_scorecard_from_controls(
                        population=population,
                        snapshot_frequency=snapshot_frequency,
                        sampling_strategy=sampling_strategy,
                    )
                except Exception as exc:
                    friendly_error(exc)
                else:
                    st.session_state["scorecard_run"] = new_run_id
                    st.success(f"Created scorecard run `{new_run_id}`")

    tab_features, tab_perf, tab_cal, tab_stability = st.tabs(
        ["Features", "Performance", "Calibration", "Stability"]
    )
    with tab_features:
        left, right = st.columns(2)
        with left:
            st.subheader("Feature List")
            st.dataframe(features, use_container_width=True, hide_index=True)
            st.subheader("WOE / IV")
            st.dataframe(iv, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Coefficients")
            st.dataframe(coefficients, use_container_width=True, hide_index=True)
            st.subheader("Score Bands")
            st.dataframe(bands, use_container_width=True, hide_index=True)
    with tab_perf:
        st.subheader("TRAIN / VALIDATION / OOT Metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if {"split", "roc_auc"}.issubset(metrics.columns):
            fig = px.bar(metrics, x="split", y="roc_auc", color="split", title="AUC by Split")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Score Distribution")
        st.dataframe(deciles, use_container_width=True, hide_index=True)
    with tab_cal:
        st.subheader("Calibration")
        if {"predicted_bad_rate", "observed_bad_rate"}.issubset(calibration.columns):
            fig = px.line(
                calibration,
                x="predicted_bad_rate",
                y="observed_bad_rate",
                color="split" if "split" in calibration.columns else None,
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(calibration, use_container_width=True, hide_index=True)
    with tab_stability:
        st.subheader("PSI")
        st.dataframe(psi, use_container_width=True, hide_index=True)
        st.subheader("Yearly Performance")
        st.dataframe(yearly, use_container_width=True, hide_index=True)


main()
