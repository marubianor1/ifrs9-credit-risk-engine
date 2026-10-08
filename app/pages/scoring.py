"""V3 model validation workbench for existing scorecard runs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.advisory import (
    apply_advisory_css,
    context_bar,
    executive_kpis,
    methodology_note,
    page_header,
    section_header,
    source_caption,
)
from app.components.charts import calibration_scatter, dumbbell_chart, finish_chart
from app.components.formatting import au_date, bps, ordered_splits, ratio
from app.components.guidance import render_guidance
from app.components.layout import friendly_error
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


def _metric_row(metrics: pd.DataFrame, split: str) -> pd.Series:
    return metrics.loc[metrics["split"].eq(split)].iloc[0]


def main() -> None:
    apply_advisory_css()
    page_header(
        "Scoring Models",
        "Model validation workbench for behavioural scorecard discrimination, calibration "
        "and stability.",
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

    run_meta = artifacts["run"]
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
    oot_split = "OOT" if "OOT" in set(metrics["split"].astype(str)) else split_order[-1]
    oot = _metric_row(metrics, oot_split)
    oot_oe = oot["observed_bad_rate"] / oot["predicted_bad_rate"]

    config = run_meta.get("config", {})
    train_window = (
        f"{config.get('train', {}).get('start', 'n/a')} to "
        f"{config.get('train', {}).get('end', 'n/a')}"
    )
    oot_window = (
        f"{config.get('oot', {}).get('start', 'n/a')} to "
        f"{config.get('oot', {}).get('end', 'n/a')}"
    )
    context_bar(
        [
            ("Model", selected),
            ("Population", str(run_meta.get("population", "behavioural")).title()),
            ("Train", train_window),
            (
                "Validation",
                f"{config.get('validation', {}).get('start', 'n/a')} to "
                f"{config.get('validation', {}).get('end', 'n/a')}",
            ),
            ("OOT", oot_window),
        ]
    )
    executive_kpis(
        [
            ("AUC", f"{oot['roc_auc']:.3f}", f"{oot_split} discrimination"),
            ("Gini", f"{oot['gini']:.3f}", "2 x AUC - 1"),
            ("KS", f"{oot['ks']:.3f}", "Maximum separation"),
            ("OOT O/E", ratio(oot_oe), "Observed default rate / predicted PD"),
        ],
        columns=4,
    )

    section_header("Calibration summary", "Observed default rate versus predicted PD by split.")
    comparison = metrics.copy()
    comparison["Difference"] = comparison["observed_bad_rate"] - comparison["predicted_bad_rate"]
    st.plotly_chart(
        dumbbell_chart(
            comparison,
            category="split",
            left_value="predicted_bad_rate",
            right_value="observed_bad_rate",
            left_label="Predicted PD",
            right_label="Observed default rate",
            title="Observed versus predicted default rate",
            xaxis_title="12-month default rate",
        ),
        width="stretch",
    )
    source_caption(f"As at {au_date('2025-03-01')} · Source: {selected}")
    st.dataframe(format_table(bad_rate_table(metrics)), width="stretch", hide_index=True)

    tab_cal, tab_disc, tab_stability, tab_features = st.tabs(
        ["Calibration", "Discrimination", "Stability", "Feature Diagnostics"]
    )
    with tab_cal:
        section_header(
            "Calibration curve",
            "Predicted PD versus observed default rate by risk band.",
        )
        if {"predicted_bad_rate", "observed_bad_rate"}.issubset(calibration.columns):
            st.plotly_chart(
                calibration_scatter(
                    calibration,
                    title="Risk-band calibration backtest",
                ),
                width="stretch",
            )
        st.dataframe(format_table(calibration), width="stretch", hide_index=True)
    with tab_disc:
        section_header("Discrimination", "Compact model performance metrics by split.")
        display_cols = ["split", "roc_auc", "gini", "ks", "pr_auc", "brier"]
        st.dataframe(format_table(metrics[display_cols]), width="stretch", hide_index=True)
    with tab_stability:
        section_header("Population stability", "PSI ranked by feature and comparison.")
        psi_ranked = psi.sort_values("psi", ascending=False).copy()
        fig = px.bar(
            psi_ranked,
            x="psi",
            y="feature",
            color="comparison",
            orientation="h",
            title="Population stability index",
            labels={"psi": "PSI", "feature": "Feature", "comparison": "Comparison"},
        )
        fig.add_vline(x=0.25, line_dash="dash", line_color="#B7791F")
        st.plotly_chart(finish_chart(fig, xaxis_title="PSI"), width="stretch")
        st.caption(
            "Dashed line is an illustrative project monitoring reference, not a "
            "regulatory threshold."
        )
        st.dataframe(format_table(psi), width="stretch", hide_index=True)
        with st.expander("Yearly diagnostics", expanded=False):
            st.dataframe(format_table(yearly), width="stretch", hide_index=True)
    with tab_features:
        left, right = st.columns(2)
        with left:
            section_header(
                "Information value",
                "Feature ranking from the existing scorecard artifacts.",
            )
            st.dataframe(format_table(iv), width="stretch", hide_index=True)
            with st.expander("Feature list", expanded=False):
                st.dataframe(format_table(features), width="stretch", hide_index=True)
        with right:
            section_header("Coefficients", "Model coefficients and sign diagnostics.")
            st.dataframe(format_table(coefficients), width="stretch", hide_index=True)
            with st.expander("Bands and deciles", expanded=False):
                st.dataframe(format_table(bands), width="stretch", hide_index=True)
                st.dataframe(format_table(deciles), width="stretch", hide_index=True)

    oot_difference = comparison.loc[comparison["split"].eq(oot_split), "Difference"].iloc[0]
    methodology_note(
        "Bad-rate differences are displayed as observed default rate less predicted PD. "
        f"For {oot_split}, the calibration difference is {bps(oot_difference)}. "
        "Monitoring statuses use project thresholds only."
    )
    render_guidance("scoring")

    with st.expander("Model settings", expanded=False):
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
            if st.button("Run Model", type="primary"):
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


main()
