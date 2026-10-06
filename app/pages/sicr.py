"""SICR and staging page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.services.runtime import full_mode_message, is_cloud_demo
from app.services.sicr import (
    load_sicr_artifacts,
    run_staging_simulation,
    staging_overrides_from_controls,
)


@st.cache_data(show_spinner=False)
def _load():
    return load_sicr_artifacts("sicr_v1_1")


def main() -> None:
    page_title("SICR & Staging", "Stage allocation, trigger diagnostics, and threshold simulation.")
    render_guidance("sicr")
    try:
        artifacts = _load()
    except Exception as exc:
        friendly_error(exc)
        return

    stage = artifacts["stage_distribution"]
    trigger = artifacts["trigger_distribution"]
    migrations = artifacts["migrations"]
    stage3 = artifacts["stage3_audit"]
    reference_lag = artifacts["reference_lag"]

    st.subheader("Baseline Stage Distribution")
    st.plotly_chart(px.bar(stage, x="stage", y="ead", color="stage"), use_container_width=True)
    st.dataframe(stage, use_container_width=True, hide_index=True)

    if is_cloud_demo():
        st.subheader("Threshold Simulation")
        st.info(full_mode_message())
    else:
        with st.form("staging_simulation"):
            st.subheader("Threshold Simulation")
            c1, c2, c3, c4, c5 = st.columns(5)
            relative_pd = c1.number_input("Relative PD", min_value=0.1, value=2.0, step=0.1)
            absolute_text = c2.text_input("Absolute PD", "")
            rating = c3.number_input("Rating downgrade", min_value=1, value=3, step=1)
            dpd = c4.number_input("DPD backstop", min_value=1, value=1, step=1)
            cure = c5.number_input("Stage 2 cure probation", min_value=0, value=3, step=1)
            run_clicked = st.form_submit_button("Run Staging Simulation", type="primary")
        if run_clicked:
            try:
                overrides = staging_overrides_from_controls(
                    relative_pd=relative_pd,
                    absolute_pd=float(absolute_text) if absolute_text else None,
                    rating_downgrade=int(rating),
                    dpd_backstop=int(dpd),
                    cure_probation=int(cure),
                )
                result = run_staging_simulation(overrides)
            except Exception as exc:
                friendly_error(exc)
            else:
                st.subheader("Simulated Stage Distribution")
                simulated = result["stage_distribution"]
                st.dataframe(simulated, use_container_width=True, hide_index=True)
                baseline = stage[["stage", "rows", "ead"]].rename(
                    columns={"rows": "baseline_rows", "ead": "baseline_ead"}
                )
                comparison = baseline.merge(
                    simulated[["stage", "rows", "ead"]].rename(
                        columns={"rows": "simulated_rows", "ead": "simulated_ead"}
                    ),
                    on="stage",
                    how="outer",
                ).fillna(0)
                comparison["row_delta"] = (
                    comparison["simulated_rows"] - comparison["baseline_rows"]
                )
                comparison["ead_delta"] = comparison["simulated_ead"] - comparison["baseline_ead"]
                st.subheader("Baseline vs Simulated")
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                st.subheader("Simulation Summary")
                st.dataframe(result["sensitivity"], use_container_width=True, hide_index=True)

    tab_triggers, tab_migrations, tab_reference = st.tabs(
        ["Triggers", "Migrations", "Reference PD"]
    )
    with tab_triggers:
        trigger_fig = px.bar(trigger, x="trigger", y="rows", color="trigger")
        st.plotly_chart(trigger_fig, use_container_width=True)
        st.dataframe(trigger, use_container_width=True, hide_index=True)
        st.subheader("Trigger Overlap")
        st.dataframe(artifacts["trigger_exclusivity"], use_container_width=True, hide_index=True)
    with tab_migrations:
        migration_fig = px.bar(migrations, x="migration", y="rows", color="migration")
        st.plotly_chart(migration_fig, use_container_width=True)
        st.dataframe(migrations, use_container_width=True, hide_index=True)
        st.subheader("Stage 3 Duration / Cures")
        st.dataframe(stage3, use_container_width=True, hide_index=True)
    with tab_reference:
        st.info(
            "Behavioural reference PD uses the earliest valid behavioural score/rating "
            "available after origination as a proxy for initial-recognition risk."
        )
        st.dataframe(reference_lag, use_container_width=True, hide_index=True)
        st.dataframe(artifacts["reference_pd"], use_container_width=True, hide_index=True)


main()
