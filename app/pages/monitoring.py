"""V3 consolidated model monitoring page."""

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
from app.components.charts import finish_chart, metric_bar, oe_bullet_chart
from app.components.formatting import SPLIT_ORDER, au_date
from app.components.guidance import render_guidance
from app.components.layout import friendly_error
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


def main() -> None:
    apply_advisory_css()
    page_header(
        "Model Monitoring",
        "Model-risk monitoring pack using transparent project thresholds.",
    )
    try:
        artifacts, thresholds = _load()
    except Exception as exc:
        friendly_error(exc)
        return

    context_bar(
        [
            ("As at", au_date("2025-03-01")),
            ("Threshold source", "config/monitoring.yaml"),
            ("Portfolio", "Freddie Mac Case Study"),
            ("Scope", "Scorecard / PD / LGD / EAD"),
        ]
    )

    scorecard = _sort_split_frame(artifacts["scorecard_metrics"])
    psi = artifacts["scorecard_psi"].copy()
    pd_backtest = _sort_split_frame(artifacts["pd_backtest"])
    lgd = _sort_split_frame(artifacts["lgd_backtest"])
    ead = _sort_split_frame(artifacts["ead_backtest"])

    matrix = _monitoring_matrix(scorecard, psi, pd_backtest, lgd, ead, thresholds)
    status_counts = matrix["Status"].value_counts().to_dict()
    executive_kpis(
        [
            ("Current monitoring status", _dominant_status(status_counts), "Project thresholds"),
            ("GREEN", str(status_counts.get("GREEN", 0)), "Within project thresholds"),
            ("AMBER", str(status_counts.get("AMBER", 0)), "Watch-list under project thresholds"),
            ("RED", str(status_counts.get("RED", 0)), "Requires review under project thresholds"),
        ],
        columns=4,
    )

    section_header(
        "Monitoring matrix",
        "Primary model-risk view by area, metric and threshold status.",
    )
    st.dataframe(
        format_table(add_status_label(matrix, column="Status")),
        width="stretch",
        hide_index=True,
    )

    tab_calibration, tab_discrimination, tab_stability, tab_detail = st.tabs(
        ["Calibration", "Discrimination", "Stability", "Detail"]
    )
    with tab_calibration:
        section_header("O/E calibration", "Observed over expected ratios centred on 1.00.")
        oe_frame = matrix[matrix["Metric"].str.contains("O/E", regex=False)].copy()
        st.plotly_chart(
            oe_bullet_chart(
                oe_frame,
                category="Metric",
                oe_value="Current",
                title="O/E against project monitoring ranges",
                thresholds=thresholds["oe_ratio"],
            ),
            width="stretch",
        )
        source_caption("As at 1 Mar 2025 · Project thresholds from config/monitoring.yaml")
    with tab_discrimination:
        section_header("Scorecard discrimination", "AUC by development split.")
        st.plotly_chart(
            metric_bar(
                scorecard,
                x="split",
                y="roc_auc",
                color="split",
                title="AUC by split",
                yaxis_title="AUC",
            ),
            width="stretch",
        )
        st.dataframe(format_table(scorecard), width="stretch", hide_index=True)
    with tab_stability:
        section_header("Population stability", "PSI ranked by feature and comparison.")
        psi_ranked = psi.sort_values("psi", ascending=False)
        fig = px.bar(
            psi_ranked,
            x="psi",
            y="feature",
            color="comparison",
            orientation="h",
            title="PSI by feature",
            labels={"psi": "PSI", "feature": "Feature", "comparison": "Comparison"},
        )
        fig.add_vline(
            x=thresholds["scorecard_psi"]["green_max"],
            line_color="#23827A",
            line_dash="dash",
        )
        fig.add_vline(
            x=thresholds["scorecard_psi"]["amber_max"],
            line_color="#B7791F",
            line_dash="dash",
        )
        st.plotly_chart(finish_chart(fig, xaxis_title="PSI"), width="stretch")
        st.caption("Reference lines are project monitoring thresholds, not regulatory thresholds.")
        st.dataframe(format_table(psi), width="stretch", hide_index=True)
    with tab_detail:
        left, right = st.columns(2)
        with left:
            section_header("PD and LGD")
            st.dataframe(
                format_table(artifacts["pd_calibration"]),
                width="stretch",
                hide_index=True,
            )
            st.dataframe(format_table(lgd), width="stretch", hide_index=True)
        with right:
            section_header("EAD and rating diagnostics")
            st.dataframe(format_table(ead), width="stretch", hide_index=True)
            with st.expander("Rating backtesting diagnostics", expanded=False):
                st.dataframe(format_table(artifacts["pd_rating"]), width="stretch", hide_index=True)

    methodology_note(
        "GREEN, AMBER and RED are configurable project monitoring thresholds. They are not APRA "
        "or regulatory thresholds. Status is shown with text plus colour to avoid colour-only "
        "coding."
    )
    render_guidance("monitoring")


def _monitoring_matrix(
    scorecard: pd.DataFrame,
    psi: pd.DataFrame,
    pd_backtest: pd.DataFrame,
    lgd: pd.DataFrame,
    ead: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    rows = []
    train_auc = _split_value(scorecard, "TRAIN", "roc_auc")
    if "roc_auc" in scorecard:
        for _, row in scorecard.iterrows():
            value = float(row["roc_auc"])
            rows.append(
                {
                    "Model area": "Scorecard",
                    "Metric": f"AUC {row.get('split', '')}",
                    "Current": value,
                    "Reference": train_auc,
                    "Change": value - train_auc if train_auc is not None else None,
                    "Threshold": _threshold_text(thresholds["scorecard_auc"], higher=True),
                    "Status": status_band(
                        value,
                        thresholds["scorecard_auc"],
                        higher_is_better=True,
                    ),
                }
            )
    max_psi = float(psi["psi"].max())
    rows.append(
        {
            "Model area": "Scorecard",
            "Metric": "Max PSI",
            "Current": max_psi,
            "Reference": 0.0,
            "Change": max_psi,
            "Threshold": _threshold_text(thresholds["scorecard_psi"], higher=False),
            "Status": status_band(
                max_psi,
                thresholds["scorecard_psi"],
                higher_is_better=False,
            ),
        }
    )
    rows.extend(_oe_rows("PD", pd_backtest, thresholds["oe_ratio"]))
    rows.extend(_oe_rows("LGD", lgd, thresholds["oe_ratio"]))
    contractual = ead[ead["method"].eq("contractual_amortization")]
    rows.extend(_oe_rows("EAD", contractual, thresholds["oe_ratio"]))
    return pd.DataFrame(rows)


def _oe_rows(area: str, frame: pd.DataFrame, threshold: dict) -> list[dict]:
    train = _split_value(frame, "TRAIN", "oe_ratio")
    rows = []
    for _, row in frame.iterrows():
        value = float(row["oe_ratio"])
        rows.append(
            {
                "Model area": area,
                "Metric": f"O/E {row['split']}",
                "Current": value,
                "Reference": train,
                "Change": value - train if train is not None else None,
                "Threshold": _threshold_text(threshold, higher=None),
                "Status": oe_status(value, threshold),
            }
        )
    return rows


def _split_value(frame: pd.DataFrame, split: str, column: str) -> float | None:
    if "split" not in frame or column not in frame:
        return None
    match = frame.loc[frame["split"].eq(split)]
    if match.empty:
        return None
    return float(match[column].iloc[0])


def _threshold_text(threshold: dict, *, higher: bool | None) -> str:
    if higher is True:
        return f"GREEN >= {threshold['green_min']:.2f}; AMBER >= {threshold['amber_min']:.2f}"
    if higher is False:
        return f"GREEN <= {threshold['green_max']:.2f}; AMBER <= {threshold['amber_max']:.2f}"
    return f"GREEN {threshold['green_min']:.2f}-{threshold['green_max']:.2f}"


def _dominant_status(status_counts: dict[str, int]) -> str:
    if status_counts.get("RED", 0):
        return "RED"
    if status_counts.get("AMBER", 0):
        return "AMBER"
    return "GREEN"


def _sort_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" not in frame.columns:
        return frame
    output = frame.copy()
    output["split"] = pd.Categorical(output["split"], categories=SPLIT_ORDER, ordered=True)
    return output.sort_values("split")


main()
