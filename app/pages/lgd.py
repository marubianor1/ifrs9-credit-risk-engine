"""LGD framework page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.charts import finish_chart, split_grouped_bar
from app.components.formatting import percentage, ratio
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import format_table
from app.components.theme import rating_order
from app.services.lgd import (
    BASELINE_RUN,
    available_lgd_runs,
    lgd_governance_status,
    load_lgd_artifacts,
    load_lgd_forward_artifacts,
)


@st.cache_data(show_spinner=False)
def _load(run_id: str):
    return load_lgd_artifacts(run_id)


@st.cache_data(show_spinner=False)
def _load_forward(run_id: str):
    return load_lgd_forward_artifacts(run_id)


def _challenger_table(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.loc[metrics["model"].str.contains("combined", case=False, na=False)]
        .pivot_table(index="model", columns="split", values="oe_ratio", aggfunc="mean")
        .reset_index()
    )
    if summary.empty:
        summary = metrics.pivot_table(
            index="model",
            columns="split",
            values="oe_ratio",
            aggfunc="mean",
        ).reset_index()
    summary["Status"] = "Rejected - temporal instability"
    summary["Reason"] = "Worse temporal / OOT stability than lgd_v1_2 baseline"
    return summary.rename(
        columns={
            "model": "Model",
            "VALIDATION": "Validation O/E",
            "OOT": "OOT O/E",
        }
    )


def main() -> None:
    page_title("LGD", "Loss given default diagnostics and governance.")
    governance = lgd_governance_status()
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

    backtesting = artifacts["backtesting"].copy()
    components = artifacts["components"].copy()
    by_rating = artifacts["by_rating"].copy()
    by_year = artifacts["by_year"].copy()
    downturn = artifacts["downturn"].copy()
    recovery = artifacts["recovery_timing"].copy()
    resolution = artifacts["resolution"].copy()
    validation = backtesting.loc[backtesting["split"].eq("VALIDATION")]
    selected_row = validation.iloc[0] if not validation.empty else backtesting.iloc[0]
    downturn_mean = float(downturn["downturn_overlay_factor"].mean())

    st.caption(f"Selected run: `{selected}` | Production baseline: `{BASELINE_RUN}`")
    status_left, status_right = st.columns([1, 1])
    with status_left:
        st.info(f"Selected baseline: `{governance['production_baseline']}`")
    with status_right:
        st.warning("Challenger status: Rejected - temporal instability")

    kpi_row(
        [
            (
                "Realised LGD",
                percentage(selected_row["realized_mean_lgd"]),
                "Validation realised mean LGD.",
            ),
            (
                "Predicted LGD",
                percentage(selected_row["predicted_lgd"]),
                "Validation predicted LGD.",
            ),
            ("O/E", ratio(selected_row["oe_ratio"]), "Observed / expected LGD."),
            (
                "Cure rate",
                percentage(selected_row["cure_rate"]),
                "Resolved cured default episodes.",
            ),
            ("Downturn sensitivity", ratio(downturn_mean), "Average downturn overlay factor."),
        ]
    )
    render_guidance("lgd")

    tab_perf, tab_components, tab_segments, tab_challengers = st.tabs(
        ["Performance", "Decomposition", "Segments", "Challengers"]
    )
    with tab_perf:
        st.subheader("Realised versus predicted LGD")
        st.plotly_chart(
            split_grouped_bar(
                backtesting,
                y=["realized_mean_lgd", "predicted_lgd"],
                title="Realised and predicted LGD by split",
                yaxis_title="LGD",
                rate_axis=True,
            ),
            width="stretch",
        )
        oe_fig = px.bar(
            backtesting,
            x="split",
            y="oe_ratio",
            color="split",
            category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
            title="LGD O/E by split",
            labels={"split": "Split", "oe_ratio": "O/E"},
            color_discrete_map={"TRAIN": "#286090", "VALIDATION": "#667085", "OOT": "#98A2B3"},
        )
        oe_fig.update_traces(showlegend=False, hovertemplate="%{x}<br>%{y:.2f}<extra></extra>")
        st.plotly_chart(finish_chart(oe_fig, yaxis_title="O/E"), width="stretch")
        st.dataframe(format_table(backtesting), width="stretch", hide_index=True)

    with tab_components:
        st.subheader("Cure / non-cure branch decomposition")
        branch = components.melt(
            id_vars=["split"],
            value_vars=[
                "observed_p_cure",
                "observed_lgd_cure",
                "observed_lgd_non_cure",
            ],
            var_name="Measure",
            value_name="Value",
        )
        branch["Measure"] = branch["Measure"].map(
            {
                "observed_p_cure": "Probability of cure",
                "observed_lgd_cure": "LGD | cure",
                "observed_lgd_non_cure": "LGD | non-cure",
            }
        )
        fig = px.bar(
            branch,
            x="split",
            y="Value",
            color="Measure",
            barmode="group",
            category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
            title="Observed cure and severity components",
            labels={"split": "Split", "Value": "Rate"},
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(finish_chart(fig, yaxis_title="Rate"), width="stretch")
        st.dataframe(format_table(components), width="stretch", hide_index=True)

    with tab_segments:
        left, right = st.columns(2)
        with left:
            st.subheader("LGD by rating")
            rating = by_rating.rename(columns={"rating_at_default": "rating"}).copy()
            rating["rating"] = pd.Categorical(rating["rating"], rating_order(), ordered=True)
            rating = rating.sort_values(["split", "rating"])
            fig = px.bar(
                rating,
                x="rating",
                y="oe_ratio",
                color="split",
                barmode="group",
                category_orders={"rating": rating_order(), "split": ["TRAIN", "VALIDATION", "OOT"]},
                title="LGD O/E by rating",
                labels={"rating": "Rating", "oe_ratio": "O/E", "split": "Split"},
            )
            st.plotly_chart(finish_chart(fig, yaxis_title="O/E"), width="stretch")
            st.dataframe(format_table(by_rating), width="stretch", hide_index=True)
        with right:
            st.subheader("LGD by default year")
            year_fig = px.line(
                by_year,
                x="default_year",
                y="oe_ratio",
                color="split",
                markers=True,
                category_orders={"split": ["TRAIN", "VALIDATION", "OOT"]},
                title="LGD O/E by default year",
                labels={"default_year": "Default year", "oe_ratio": "O/E", "split": "Split"},
            )
            st.plotly_chart(
                finish_chart(year_fig, yaxis_title="O/E", xaxis_title="Default year"),
                width="stretch",
            )
            st.dataframe(format_table(by_year), width="stretch", hide_index=True)
        st.subheader("Downturn sensitivity")
        downturn_chart = downturn.copy()
        downturn_chart["delta"] = downturn_chart["downturn_overlay_factor"] - 1.0
        fig = px.bar(
            downturn_chart,
            x="rating_at_default",
            y="delta",
            title="Downturn LGD overlay above baseline",
            labels={"rating_at_default": "Rating", "delta": "Overlay above baseline"},
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(
            finish_chart(fig, yaxis_title="Overlay above baseline"),
            width="stretch",
        )
        with st.expander("Recovery timing and resolution population", expanded=False):
            st.dataframe(format_table(recovery), width="stretch", hide_index=True)
            st.dataframe(format_table(resolution), width="stretch", hide_index=True)

    with tab_challengers:
        st.info(
            "LGD challenger runs are visible for governance review only. "
            "The UI does not refit LGD."
        )
        st.dataframe(
            format_table(_challenger_table(artifacts["model_metrics"])),
            width="stretch",
            hide_index=True,
        )
        fl_run = st.selectbox("Forward-looking LGD challenger", ["lgd_fl_v1", "lgd_fl_v2"])
        try:
            fl_artifacts = _load_forward(fl_run)
        except Exception as exc:
            friendly_error(exc)
        else:
            with st.expander("Forward-looking challenger diagnostics", expanded=False):
                st.dataframe(
                    format_table(fl_artifacts["weighted"]),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    format_table(fl_artifacts["scenario_by_rating"]),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    format_table(fl_artifacts["diagnostics"]),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    format_table(fl_artifacts["sensitivity"]),
                    width="stretch",
                    hide_index=True,
                )


main()
