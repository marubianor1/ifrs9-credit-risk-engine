"""Consolidated model monitoring page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.layout import friendly_error, page_title
from app.services.monitoring import (
    load_monitoring_artifacts,
    load_monitoring_thresholds,
    oe_status,
    status_band,
)


@st.cache_data(show_spinner=False)
def _load():
    return load_monitoring_artifacts(), load_monitoring_thresholds()


def _status_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    page_title("Model Monitoring", "Project monitoring thresholds and diagnostic artifacts.")
    try:
        artifacts, thresholds = _load()
    except Exception as exc:
        friendly_error(exc)
        return

    st.info("Statuses use project monitoring thresholds from `config/monitoring.yaml`.")
    scorecard = artifacts["scorecard_metrics"]
    psi = artifacts["scorecard_psi"]
    pd_backtest = artifacts["pd_backtest"]
    lgd = artifacts["lgd_backtest"]
    ead = artifacts["ead_backtest"]

    rows = []
    if "roc_auc" in scorecard:
        for _, row in scorecard.iterrows():
            rows.append(
                {
                    "area": "Scorecard",
                    "metric": f"AUC {row.get('split', '')}",
                    "value": row["roc_auc"],
                    "status": status_band(
                        row["roc_auc"],
                        thresholds["scorecard_auc"],
                        higher_is_better=True,
                    ),
                }
            )
    rows.append(
        {
            "area": "Scorecard",
            "metric": "Max PSI",
            "value": psi["psi"].max(),
            "status": status_band(
                psi["psi"].max(),
                thresholds["scorecard_psi"],
                higher_is_better=False,
            ),
        }
    )
    for _, row in pd_backtest.iterrows():
        rows.append(
            {
                "area": "PD",
                "metric": f"O/E {row['split']}",
                "value": row["oe_ratio"],
                "status": oe_status(row["oe_ratio"], thresholds["oe_ratio"]),
            }
        )
    for _, row in lgd.iterrows():
        rows.append(
            {
                "area": "LGD",
                "metric": f"O/E {row['split']}",
                "value": row["oe_ratio"],
                "status": oe_status(row["oe_ratio"], thresholds["oe_ratio"]),
            }
        )
    contractual = ead[ead["method"].eq("contractual_amortization")]
    for _, row in contractual.iterrows():
        rows.append(
            {
                "area": "EAD",
                "metric": f"O/E {row['split']}",
                "value": row["oe_ratio"],
                "status": oe_status(row["oe_ratio"], thresholds["oe_ratio"]),
            }
        )

    status = _status_table(rows)
    st.subheader("Traffic-light Summary")
    st.dataframe(status, use_container_width=True, hide_index=True)

    tab_scorecard, tab_pd, tab_lgd, tab_ead = st.tabs(["Scorecard", "PD", "LGD", "EAD"])
    with tab_scorecard:
        scorecard_fig = px.bar(scorecard, x="split", y="roc_auc", color="split")
        st.plotly_chart(scorecard_fig, use_container_width=True)
        st.dataframe(scorecard, use_container_width=True, hide_index=True)
        st.subheader("PSI")
        st.dataframe(psi, use_container_width=True, hide_index=True)
        st.subheader("Yearly Performance")
        st.dataframe(artifacts["scorecard_yearly"], use_container_width=True, hide_index=True)
    with tab_pd:
        st.dataframe(artifacts["pd_calibration"], use_container_width=True, hide_index=True)
        st.dataframe(pd_backtest, use_container_width=True, hide_index=True)
        st.dataframe(artifacts["pd_rating"], use_container_width=True, hide_index=True)
    with tab_lgd:
        st.dataframe(lgd, use_container_width=True, hide_index=True)
        st.dataframe(artifacts["lgd_components"], use_container_width=True, hide_index=True)
    with tab_ead:
        st.dataframe(ead, use_container_width=True, hide_index=True)


main()
