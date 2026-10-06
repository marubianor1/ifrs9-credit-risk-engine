"""PD framework page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.charts import finish_chart, rating_bar
from app.components.formatting import percentage, ratio
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
from app.components.theme import rating_order
from app.services.pd import available_pd_runs, load_pd_artifacts, run_pd_from_controls
from app.services.runtime import full_mode_message, is_cloud_demo


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_pd_artifacts(run_id)


def main() -> None:
    page_title("PD", "Calibration, rating scale, lifetime PD and backtesting artifacts.")
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

    metrics = artifacts["calibration_metrics"].copy()
    rating_summary = artifacts["rating_summary"].copy()
    rating_ttc = artifacts["rating_ttc"].copy()
    backtest = artifacts["backtesting_split"].copy()
    lifetime = artifacts["lifetime"].copy()
    pit_ttc = artifacts["pit_ttc"].copy()
    oot = backtest.loc[backtest["split"].eq("OOT")]
    portfolio_pd = float(rating_summary["mean_pd"].mean())
    ttc_pd = float(rating_ttc["rating_ttc_pd"].mean())
    calibration_method = str(metrics["pd_type"].iloc[0]).replace("_", " ").title()
    oot_oe = float(oot["oe_ratio"].iloc[0]) if not oot.empty else None

    st.caption(f"Selected run: `{selected}` | Reporting date: 1 Mar 2025")
    kpi_row(
        [
            ("Portfolio calibrated PD", percentage(portfolio_pd), "Average calibrated PD."),
            ("TTC PD", percentage(ttc_pd), "Through-the-cycle reference PD."),
            ("Calibration method", calibration_method, "Configured calibration approach."),
            (
                "Active rating grades",
                f"{rating_summary['rating'].nunique():,.0f}",
                "Grades in use.",
            ),
            (
                "OOT O/E",
                ratio(oot_oe) if oot_oe is not None else "n/a",
                "Out-of-time observed / expected.",
            ),
        ]
    )
    render_guidance("pd")

    tab_cal, tab_rating, tab_lifetime, tab_forward, tab_backtest = st.tabs(
        ["Calibration", "Ratings", "Lifetime PD", "Forward-looking", "Backtesting"]
    )
    with tab_cal:
        st.subheader("Raw versus calibrated PD")
        chart = metrics.loc[metrics["pd_type"].isin(["raw", "calibrated"])].copy()
        if not chart.empty:
            chart["Model PD"] = chart["pd_type"].map(
                {"raw": "Raw PD", "calibrated": "Calibrated PD"}
            )
            chart = chart.rename(
                columns={
                    "observed_bad_rate": "Observed default rate",
                    "predicted_bad_rate": "Predicted PD",
                }
            )
            plot_data = chart.melt(
                id_vars=["split", "Model PD"],
                value_vars=["Observed default rate", "Predicted PD"],
                var_name="Measure",
                value_name="Rate",
            )
            fig = px.bar(
                plot_data,
                x="split",
                y="Rate",
                color="Measure",
                facet_col="Model PD",
                barmode="group",
                title="Observed default rate versus model PD by split",
                category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
                labels={"split": "Split", "Rate": "Rate"},
                color_discrete_map={
                    "Observed default rate": "#286090",
                    "Predicted PD": "#287C8E",
                },
            )
            fig.update_yaxes(tickformat=".1%")
            fig.update_traces(hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
            st.plotly_chart(finish_chart(fig, yaxis_title="Rate"), width="stretch")
        st.dataframe(format_table(metrics), width="stretch", hide_index=True)

        curve = artifacts["calibration_curve"].copy()
        if {"predicted_bad_rate", "observed_bad_rate"}.issubset(curve.columns):
            scatter = px.scatter(
                curve,
                x="predicted_bad_rate",
                y="observed_bad_rate",
                color="split",
                hover_data={
                    "bucket": True,
                    "rows": ":,",
                    "predicted_bad_rate": ":.2%",
                    "observed_bad_rate": ":.2%",
                },
                title="Calibration by risk band",
                category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
                labels={
                    "predicted_bad_rate": "Predicted PD",
                    "observed_bad_rate": "Observed default rate",
                    "split": "Split",
                },
            )
            max_value = max(curve["predicted_bad_rate"].max(), curve["observed_bad_rate"].max())
            scatter.add_scatter(
                x=[0, max_value],
                y=[0, max_value],
                mode="lines",
                name="Observed = Predicted",
                line={"dash": "dash", "color": "#667085"},
            )
            scatter.update_xaxes(tickformat=".1%")
            scatter.update_yaxes(tickformat=".1%")
            st.plotly_chart(
                finish_chart(
                    scatter,
                    yaxis_title="Observed default rate",
                    xaxis_title="Predicted PD",
                ),
                width="stretch",
            )

    with tab_rating:
        left, right = st.columns(2)
        with left:
            st.subheader("Rating master scale")
            ordered = rating_summary.sort_values("rating")
            st.plotly_chart(
                rating_bar(
                    ordered,
                    value="mean_pd",
                    title="Calibrated PD by rating grade",
                    yaxis_title="Calibrated PD",
                ),
                width="stretch",
            )
            st.dataframe(format_table(ordered), width="stretch", hide_index=True)
        with right:
            st.subheader("TTC PD")
            ttc = rating_ttc.sort_values("rating")
            fig = px.bar(
                ttc,
                x="rating",
                y="rating_ttc_pd",
                color="rating",
                category_orders={"rating": rating_order()},
                title="Through-the-cycle PD by rating",
                labels={"rating": "Rating", "rating_ttc_pd": "TTC PD"},
            )
            fig.update_yaxes(tickformat=".1%")
            fig.update_traces(showlegend=False, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
            st.plotly_chart(finish_chart(fig, yaxis_title="TTC PD"), width="stretch")
            st.dataframe(format_table(ttc), width="stretch", hide_index=True)
            with st.expander("Rating boundaries", expanded=False):
                st.dataframe(
                    format_table(artifacts["rating_boundaries"]),
                    width="stretch",
                    hide_index=True,
                )

    with tab_lifetime:
        available = [rating for rating in rating_order() if rating in set(lifetime["rating"])]
        ratings = st.multiselect("Ratings", available, default=available[:3])
        filtered = lifetime[lifetime["rating"].isin(ratings)]
        cumulative = px.line(
            filtered,
            x="month",
            y="cumulative_pd",
            color="rating",
            category_orders={"rating": rating_order()},
            title="Lifetime cumulative PD by rating",
            labels={"month": "Month", "cumulative_pd": "Cumulative PD", "rating": "Rating"},
        )
        cumulative.update_yaxes(tickformat=".1%")
        st.plotly_chart(
            finish_chart(cumulative, yaxis_title="Cumulative PD", xaxis_title="Month"),
            width="stretch",
        )
        marginal = filtered[filtered["month"].le(60)]
        marginal_fig = px.line(
            marginal,
            x="month",
            y="marginal_pd",
            color="rating",
            category_orders={"rating": rating_order()},
            title="Monthly marginal PD by rating, first 60 months",
            labels={"month": "Month", "marginal_pd": "Marginal PD", "rating": "Rating"},
        )
        marginal_fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(
            finish_chart(marginal_fig, yaxis_title="Marginal PD", xaxis_title="Month"),
            width="stretch",
        )

    with tab_forward:
        st.subheader("PIT / TTC diagnostics")
        pit = pit_ttc.melt(
            id_vars=["observation_year"],
            value_vars=["observed_12m_default_rate", "mean_calibrated_pd", "ttc_pd"],
            var_name="Measure",
            value_name="Rate",
        )
        pit["Measure"] = pit["Measure"].map(
            {
                "observed_12m_default_rate": "Observed default rate",
                "mean_calibrated_pd": "Calibrated PD",
                "ttc_pd": "TTC PD",
            }
        )
        fig = px.line(
            pit,
            x="observation_year",
            y="Rate",
            color="Measure",
            title="Observed, PIT calibrated and TTC PD",
            labels={"observation_year": "Observation year", "Rate": "Rate"},
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(
            finish_chart(fig, yaxis_title="Rate", xaxis_title="Observation year"),
            width="stretch",
        )
        st.dataframe(format_table(pit_ttc), width="stretch", hide_index=True)

    with tab_backtest:
        st.subheader("Backtesting by split")
        st.dataframe(format_table(backtest), width="stretch", hide_index=True)
        st.subheader("Yearly rating backtesting")
        yearly = artifacts["backtesting_year"]
        fig = px.line(
            yearly,
            x="observation_year",
            y="oe_ratio",
            color="rating",
            markers=True,
            category_orders={"rating": rating_order()},
            title="O/E by rating and observation year",
            labels={
                "observation_year": "Observation year",
                "oe_ratio": "O/E",
                "rating": "Rating",
            },
        )
        st.plotly_chart(
            finish_chart(fig, yaxis_title="O/E", xaxis_title="Observation year"),
            width="stretch",
        )
        st.subheader("Transition matrix")
        transition = artifacts["transition"].pivot_table(
            index="rating_from",
            columns="to_state",
            values="probability",
            aggfunc="sum",
            fill_value=0.0,
        )
        heatmap = px.imshow(
            transition,
            text_auto=".1%",
            color_continuous_scale="Blues",
            title="Rating transition probability matrix",
            labels={"x": "To state", "y": "From rating", "color": "Probability"},
        )
        st.plotly_chart(finish_chart(heatmap), width="stretch")
        st.dataframe(
            format_table(artifacts["transition_summary"]),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Future run controls", expanded=False):
        if is_cloud_demo():
            st.info(full_mode_message())
        else:
            c1, c2, c3, c4 = st.columns(4)
            method = c1.selectbox("Calibration method", ["split_mean_anchor", "isotonic", "platt"])
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
                _ = (method, anchor_dates, rating_config)
                with st.spinner("Running PD framework via existing backend API..."):
                    try:
                        new_run_id = run_pd_from_controls(
                            scorecard_run_id=st.session_state.get(
                                "scorecard_run",
                                "behavioural_qe_v1",
                            ),
                            max_lifetime_horizon=int(horizon),
                        )
                    except Exception as exc:
                        friendly_error(exc)
                    else:
                        st.session_state["pd_run"] = new_run_id
                        st.success(f"Created PD run `{new_run_id}`")


main()
