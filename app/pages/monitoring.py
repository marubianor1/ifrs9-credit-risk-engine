"""Consolidated model monitoring page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.charts import metric_bar
from app.components.formatting import SPLIT_ORDER, display_label
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, page_title
from app.components.metrics import kpi_row
from app.components.tables import add_status_label, format_table
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
    page_title(
        "Model Monitoring",
        "Model-risk dashboard using transparent project thresholds for status assessment.",
    )
    render_guidance("monitoring")
    try:
        artifacts, thresholds = _load()
    except Exception as exc:
        friendly_error(exc)
        return

    st.caption("Threshold source: `config/monitoring.yaml` | Reporting date: 1 Mar 2025")
    scorecard = artifacts["scorecard_metrics"]
    psi = artifacts["scorecard_psi"]
    pd_backtest = artifacts["pd_backtest"]
    lgd = artifacts["lgd_backtest"]
    ead = artifacts["ead_backtest"]
    scorecard = _sort_split_frame(scorecard)
    pd_backtest = _sort_split_frame(pd_backtest)
    lgd = _sort_split_frame(lgd)
    ead = _sort_split_frame(ead)

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
    status_counts = status["status"].value_counts().to_dict()
    kpi_row(
        [
            ("GREEN", str(status_counts.get("GREEN", 0)), "Within project monitoring thresholds."),
            ("AMBER", str(status_counts.get("AMBER", 0)), "Watch-list under project thresholds."),
            ("RED", str(status_counts.get("RED", 0)), "Requires review under project thresholds."),
        ]
    )
    st.subheader("Monitoring status by area")
    st.dataframe(
        add_status_label(format_status_values(status), column="Status"),
        width="stretch",
        hide_index=True,
    )

    tab_scorecard, tab_pd, tab_lgd, tab_ead = st.tabs(["Scorecard", "PD", "LGD", "EAD"])
    with tab_scorecard:
        st.plotly_chart(
            metric_bar(
                scorecard,
                x="split",
                y="roc_auc",
                color="split",
                title="Scorecard discrimination by split",
                yaxis_title="AUC",
            ),
            width="stretch",
        )
        st.dataframe(format_table(scorecard), width="stretch", hide_index=True)
        with st.expander("PSI and yearly performance diagnostics", expanded=True):
            st.dataframe(format_table(psi), width="stretch", hide_index=True)
            st.dataframe(
                format_table(artifacts["scorecard_yearly"]),
                width="stretch",
                hide_index=True,
            )
    with tab_pd:
        st.subheader("PD calibration and O/E")
        st.dataframe(
            format_table(artifacts["pd_calibration"]),
            width="stretch",
            hide_index=True,
        )
        st.dataframe(format_table(pd_backtest), width="stretch", hide_index=True)
        with st.expander("Rating backtesting diagnostics", expanded=False):
            st.dataframe(
                format_table(artifacts["pd_rating"]),
                width="stretch",
                hide_index=True,
            )
    with tab_lgd:
        st.subheader("LGD realized versus predicted")
        st.dataframe(format_table(lgd), width="stretch", hide_index=True)
        with st.expander("Cure and severity decomposition", expanded=False):
            st.dataframe(
                format_table(artifacts["lgd_components"]),
                width="stretch",
                hide_index=True,
            )
    with tab_ead:
        st.subheader("EAD O/E, MAE and RMSE")
        st.dataframe(format_table(ead), width="stretch", hide_index=True)


def format_status_values(status: pd.DataFrame) -> pd.DataFrame:
    """Prepare status values for display without relying on colour alone."""
    output = status.copy()
    output["value"] = output["value"].map(lambda value: f"{float(value):,.2f}")
    return output.rename(columns={column: display_label(column) for column in output.columns})


def _sort_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" not in frame.columns:
        return frame
    output = frame.copy()
    output["split"] = pd.Categorical(output["split"], categories=SPLIT_ORDER, ordered=True)
    return output.sort_values("split")


main()
