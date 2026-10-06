"""SICR and staging page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.charts import finish_chart, stage_bar, stage_mix_bar
from app.components.formatting import percentage, usd
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
from app.services.runtime import full_mode_message, is_cloud_demo
from app.services.sicr import (
    load_sicr_artifacts,
    run_staging_simulation,
    staging_overrides_from_controls,
)


@st.cache_data(show_spinner=False)
def _load():
    return load_sicr_artifacts("sicr_v1_1")


def _stage3_value(stage3: pd.DataFrame, pattern: str) -> float:
    row = stage3[stage3["metric"].astype(str).str.contains(pattern, case=False, na=False)]
    return float(row["value"].iloc[0]) if not row.empty else 0.0


def main() -> None:
    page_title("SICR & Staging", "IFRS 9 stage allocation, trigger diagnostics and governance.")
    try:
        artifacts = _load()
    except Exception as exc:
        friendly_error(exc)
        return

    stage = artifacts["stage_distribution"].copy()
    trigger = artifacts["trigger_distribution"].copy()
    migrations = artifacts["migrations"].copy()
    stage3 = artifacts["stage3_audit"].copy()
    reference_lag = artifacts["reference_lag"].copy()
    total_ead = float(stage["ead"].sum())
    stage_23_ead = float(stage.loc[stage["stage"].isin([2, 3]), "ead"].sum())
    active_defaults = _stage3_value(stage3, "active")

    st.caption("Selected run: `sicr_v1_1` | Stage 3 uses active default state")
    kpi_row(
        [
            (
                "Stage 1 exposure",
                usd(stage.loc[stage["stage"].eq(1), "ead"].sum()),
                "Performing EAD.",
            ),
            ("Stage 2 exposure", usd(stage.loc[stage["stage"].eq(2), "ead"].sum()), "SICR EAD."),
            (
                "Stage 3 exposure",
                usd(stage.loc[stage["stage"].eq(3), "ead"].sum()),
                "Credit-impaired EAD.",
            ),
            (
                "Stage 2 + 3 share",
                percentage(stage_23_ead / total_ead),
                "Higher-risk exposure share.",
            ),
            ("Active defaults", f"{active_defaults:,.0f}", "Active Stage 3 default-state rows."),
        ]
    )
    render_guidance("sicr")

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(
            stage_mix_bar(stage.rename(columns={"ead": "total_ead"}), title="Stage exposure mix"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            stage_bar(
                stage.rename(columns={"ead": "total_ead"}),
                value="total_ead",
                title="Stage EAD",
                yaxis_title="EAD",
            ),
            width="stretch",
        )

    tab_triggers, tab_migrations, tab_reference, tab_sim = st.tabs(
        ["Triggers", "Migrations", "Reference PD", "Simulation"]
    )
    with tab_triggers:
        st.subheader("SICR trigger contributions")
        trigger_fig = px.bar(
            trigger,
            x="trigger",
            y="ead",
            title="Trigger EAD contributions, including overlaps",
            labels={"trigger": "Trigger", "ead": "EAD"},
            color_discrete_sequence=["#286090"],
        )
        st.plotly_chart(finish_chart(trigger_fig, yaxis_title="EAD"), width="stretch")
        st.dataframe(format_table(trigger), width="stretch", hide_index=True)
        st.subheader("Trigger overlap")
        overlap = artifacts["trigger_exclusivity"].copy()
        overlap_fig = px.bar(
            overlap,
            x="trigger_set",
            y="ead",
            title="Overlapping SICR trigger sets",
            labels={"trigger_set": "Trigger overlap", "ead": "EAD"},
            color_discrete_sequence=["#667085"],
        )
        st.plotly_chart(finish_chart(overlap_fig, yaxis_title="EAD"), width="stretch")
        st.dataframe(format_table(overlap), width="stretch", hide_index=True)

    with tab_migrations:
        st.subheader("Stage migration matrix")
        matrix = migrations.copy()
        split = matrix["migration"].astype(str).str.split("->", expand=True)
        if split.shape[1] == 2:
            matrix["From stage"] = split[0].str.strip()
            matrix["To stage"] = split[1].str.strip()
            pivot = matrix.pivot_table(
                index="From stage",
                columns="To stage",
                values="ead",
                aggfunc="sum",
                fill_value=0.0,
            )
            heatmap = px.imshow(
                pivot,
                text_auto=".2s",
                color_continuous_scale="Blues",
                title="Stage migration by EAD",
                labels={"x": "To stage", "y": "From stage", "color": "EAD"},
            )
            st.plotly_chart(finish_chart(heatmap), width="stretch")
        st.dataframe(format_table(migrations), width="stretch", hide_index=True)
        st.subheader("Stage 3 active-state audit")
        st.dataframe(format_table(stage3), width="stretch", hide_index=True)

    with tab_reference:
        st.info(
            "Behavioural reference PD uses the earliest valid behavioural score/rating available "
            "after origination as a proxy for initial-recognition risk."
        )
        st.dataframe(format_table(reference_lag), width="stretch", hide_index=True)
        st.dataframe(
            format_table(artifacts["reference_pd"]),
            width="stretch",
            hide_index=True,
        )

    with tab_sim:
        if is_cloud_demo():
            st.info(full_mode_message())
        else:
            with st.form("staging_simulation"):
                st.subheader("Threshold simulation")
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
                    st.dataframe(
                        format_table(result["stage_distribution"]),
                        width="stretch",
                        hide_index=True,
                    )
                    st.dataframe(
                        format_table(result["sensitivity"]),
                        width="stretch",
                        hide_index=True,
                    )


main()
