"""EAD framework page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.charts import finish_chart
from app.components.formatting import ratio, usd
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
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
    page_title("EAD", "Exposure at default diagnostics for amortising Freddie Mac mortgages.")
    runs = available_ead_runs()
    selected = st.selectbox("EAD run", runs, index=runs.index("ead_v1") if "ead_v1" in runs else 0)
    try:
        artifacts = _load(selected)
    except Exception as exc:
        friendly_error(exc)
        return

    ratio_dist = artifacts["ratio_distribution"]
    split = artifacts["backtest_split"].copy()
    segment = artifacts["backtest_segment"].copy()
    profiles = artifacts["profiles"]
    baseline = split.loc[split["method"].eq(BASELINE_METHOD)]
    oot = baseline.loc[baseline["split"].eq("OOT")]
    selected_row = oot.iloc[0] if not oot.empty else baseline.iloc[0]

    st.caption(f"Selected run: `{selected}` | Selected baseline: Contractual amortisation")
    kpi_row(
        [
            ("Selected method", "Contractual amortisation", "Baseline EAD method."),
            ("Observed EAD", usd(selected_row["actual_mean_ead"]), "Mean realised EAD."),
            ("Predicted EAD", usd(selected_row["predicted_mean_ead"]), "Mean predicted EAD."),
            ("O/E", ratio(selected_row["oe_ratio"]), "Observed / expected EAD."),
            ("MAE", usd(selected_row["mae"]), "Mean absolute error."),
            ("RMSE", usd(selected_row["rmse"]), "Root mean squared error."),
        ]
    )
    render_guidance("ead")

    tab_methods, tab_segments, tab_profiles = st.tabs(["Method Comparison", "Segments", "Profiles"])
    with tab_methods:
        st.subheader("Method comparison")
        oe_fig = px.bar(
            split,
            x="split",
            y="oe_ratio",
            color="method",
            barmode="group",
            category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
            title="EAD O/E by method and split",
            labels={"split": "Split", "oe_ratio": "O/E", "method": "Method"},
        )
        st.plotly_chart(finish_chart(oe_fig, yaxis_title="O/E"), use_container_width=True)
        errors = split.melt(
            id_vars=["method", "split"],
            value_vars=["mae", "rmse"],
            var_name="Metric",
            value_name="Error",
        )
        errors["Metric"] = errors["Metric"].str.upper()
        error_fig = px.bar(
            errors,
            x="split",
            y="Error",
            color="Metric",
            facet_col="method",
            barmode="group",
            category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
            title="MAE and RMSE by method",
            labels={"split": "Split", "Error": "Prediction error"},
        )
        st.plotly_chart(
            finish_chart(error_fig, yaxis_title="Prediction error"),
            use_container_width=True,
        )
        st.dataframe(format_table(split), use_container_width=True, hide_index=True)

    with tab_segments:
        mtd = st.selectbox(
            "Method for segment view",
            sorted(segment["method"].unique()),
            index=sorted(segment["method"].unique()).index(BASELINE_METHOD)
            if BASELINE_METHOD in set(segment["method"])
            else 0,
        )
        filtered = segment[segment["method"].eq(mtd)]
        segment_fig = px.bar(
            filtered,
            x="segment_value",
            y="oe_ratio",
            color="split",
            barmode="group",
            category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
            title=f"O/E by segment for {mtd.replace('_', ' ').title()}",
            labels={"segment_value": "Segment", "oe_ratio": "O/E", "split": "Split"},
        )
        st.plotly_chart(finish_chart(segment_fig, yaxis_title="O/E"), use_container_width=True)
        st.dataframe(format_table(filtered), use_container_width=True, hide_index=True)

    with tab_profiles:
        sample = profiles.head(2000)
        value_cols = [col for col in sample.columns if col.startswith("ead_")]
        if value_cols:
            profile = sample.groupby("month", observed=True)[value_cols].mean().reset_index()
            profile = profile.rename(
                columns={
                    "ead_current_balance": "Current balance",
                    "ead_contractual": "Contractual amortisation",
                    "ead_modelled": "Empirical model",
                }
            )
            fig = px.line(
                profile,
                x="month",
                y=["Current balance", "Contractual amortisation", "Empirical model"],
                title="Lifetime EAD profile",
                labels={"month": "Month", "value": "EAD", "variable": "Method"},
            )
            st.plotly_chart(
                finish_chart(fig, yaxis_title="EAD", xaxis_title="Month"),
                use_container_width=True,
            )
        st.dataframe(
            format_table(artifacts["profile_validation"]),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("EAD ratio distribution", expanded=False):
            st.dataframe(format_table(ratio_dist), use_container_width=True, hide_index=True)

    with st.expander("Future EAD run controls", expanded=False):
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
            amortization = c3.selectbox(
                "Amortization assumption",
                ["standard", "zero_interest_safe"],
            )
            if st.button("Run EAD", type="primary"):
                _ = (method, max_horizon, amortization)
                with st.spinner("Running EAD framework via existing backend API..."):
                    try:
                        new_run = run_ead_from_controls()
                    except Exception as exc:
                        friendly_error(exc)
                    else:
                        st.success(f"Created EAD run `{new_run}`")


main()
