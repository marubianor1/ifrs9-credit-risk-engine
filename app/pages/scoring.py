"""Scoring model page for existing scorecard runs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.charts import calibration_scatter, finish_chart, metric_bar
from app.components.formatting import SPLIT_ORDER, ordered_splits, percentage
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import bad_rate_table, format_table
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
    page_title(
        "Scoring Models",
        "Scorecard discrimination, calibration and stability diagnostics for model-risk review.",
    )
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
    split_order = ordered_splits(metrics["split"].tolist())
    metrics = metrics.assign(
        split=pd.Categorical(metrics["split"], categories=split_order, ordered=True)
    ).sort_values("split")

    st.caption(f"Selected run: `{selected}` | Reporting date: 1 Mar 2025")
    render_guidance("scoring")
    selected_split = st.selectbox(
        "Performance split",
        split_order,
        index=split_order.index("OOT") if "OOT" in split_order else 0,
    )
    split_row = metrics.loc[metrics["split"].eq(selected_split)].iloc[0]
    kpi_row(
        [
            ("AUC", f"{split_row['roc_auc']:.3f}", "Discrimination on the selected split."),
            ("Gini", f"{split_row['gini']:.3f}", "2 x AUC - 1."),
            ("KS", f"{split_row['ks']:.3f}", "Maximum separation between goods and bads."),
            (
                "Observed default rate",
                percentage(split_row["observed_bad_rate"]),
                "12-month observed default rate.",
            ),
            (
                "Predicted PD",
                percentage(split_row["predicted_bad_rate"]),
                "Mean predicted probability of default.",
            ),
        ]
    )

    with st.expander("Future run controls", expanded=False):
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

    table_left, chart_right = st.columns([1.2, 1])
    with table_left:
        st.subheader("12-month observed default rate versus predicted PD")
        st.dataframe(
            format_table(bad_rate_table(metrics)),
            width="stretch",
            hide_index=True,
        )
    with chart_right:
        comparison = metrics.melt(
            id_vars=["split"],
            value_vars=["observed_bad_rate", "predicted_bad_rate"],
            var_name="measure",
            value_name="rate",
        )
        comparison["measure"] = comparison["measure"].map(
            {
                "observed_bad_rate": "Observed default rate",
                "predicted_bad_rate": "Predicted PD",
            }
        )
        fig = px.bar(
            comparison,
            x="split",
            y="rate",
            color="measure",
            barmode="group",
            title="Observed default rate and predicted PD by split",
            category_orders={"split": SPLIT_ORDER},
            labels={"split": "Split", "rate": "Rate", "measure": "Measure"},
            color_discrete_map={
                "Observed default rate": "#286090",
                "Predicted PD": "#287C8E",
            },
        )
        fig.update_traces(hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(finish_chart(fig, yaxis_title="Rate"), width="stretch")

    tab_perf, tab_cal, tab_stability, tab_features = st.tabs(
        ["Performance", "Calibration", "Stability", "Feature Diagnostics"]
    )
    with tab_perf:
        render_guidance("scoring_performance")
        st.plotly_chart(
            metric_bar(
                metrics,
                x="split",
                y="roc_auc",
                color="split",
                title="AUC by development split",
                yaxis_title="AUC",
            ),
            width="stretch",
        )
        st.dataframe(format_table(metrics), width="stretch", hide_index=True)
    with tab_cal:
        st.subheader("Calibration curve")
        render_guidance("scoring_calibration")
        if {"predicted_bad_rate", "observed_bad_rate"}.issubset(calibration.columns):
            st.plotly_chart(
                calibration_scatter(
                    calibration,
                    title="Calibration backtest: observed default rate versus predicted PD",
                ),
                width="stretch",
            )
        st.dataframe(format_table(calibration), width="stretch", hide_index=True)
    with tab_stability:
        st.plotly_chart(
            metric_bar(
                psi,
                x="feature",
                y="psi",
                color="comparison",
                title="Population stability index by feature",
                yaxis_title="PSI",
            ),
            width="stretch",
        )
        st.dataframe(format_table(psi), width="stretch", hide_index=True)
        with st.expander("Yearly performance diagnostics", expanded=False):
            st.dataframe(format_table(yearly), width="stretch", hide_index=True)
    with tab_features:
        left, right = st.columns(2)
        with left:
            st.subheader("Information value ranking")
            st.dataframe(format_table(iv), width="stretch", hide_index=True)
            with st.expander("Feature list", expanded=False):
                st.dataframe(format_table(features), width="stretch", hide_index=True)
        with right:
            st.subheader("Coefficients")
            st.dataframe(format_table(coefficients), width="stretch", hide_index=True)
            with st.expander("Score bands and deciles", expanded=False):
                st.dataframe(format_table(bands), width="stretch", hide_index=True)
                st.dataframe(format_table(deciles), width="stretch", hide_index=True)


main()
