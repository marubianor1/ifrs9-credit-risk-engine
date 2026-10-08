"""V3 ECL provisioning page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.advisory import (
    apply_advisory_css,
    context_bar,
    executive_kpis,
    methodology_note,
    page_header,
    section_header,
    source_caption,
    stage_concentration_table,
)
from app.components.charts import (
    dumbbell_chart,
    horizontal_contribution_bar,
    scenario_delta_bar,
)
from app.components.formatting import au_date, percentage, signed_usd, usd
from app.components.guidance import render_guidance
from app.components.layout import friendly_error
from app.components.tables import format_table
from app.services.ecl import ecl_kpis, load_ecl_artifacts, run_ecl_from_ui
from app.services.runtime import full_mode_message, is_cloud_demo


@st.cache_data(show_spinner=False)
def _load():
    return load_ecl_artifacts("ecl_v1")


def _downturn_comparison(downturn: pd.DataFrame) -> pd.DataFrame:
    structural = downturn.loc[downturn["lgd_method"].eq("structural_base")].iloc[0]
    stressed = downturn.loc[downturn["lgd_method"].eq("downturn_sensitivity")].iloc[0]
    return pd.DataFrame(
        [
            {
                "Metric": "Weighted ECL",
                "Structural": float(structural["weighted_ecl"]),
                "Downturn": float(stressed["weighted_ecl"]),
            }
        ]
    )


def main() -> None:
    apply_advisory_css()
    page_header(
        "Expected Credit Loss",
        "Provisioning mechanics, concentration and sensitivity from the validated ECL engine.",
    )
    try:
        artifacts = _load()
    except Exception as exc:
        friendly_error(exc)
        return
    stage = artifacts["stage"]
    rating = artifacts["rating"]
    scenario = artifacts["scenario"]
    downturn = artifacts["downturn"]
    kpis = ecl_kpis(stage, scenario)
    stage_ecl = {int(row["stage"]): float(row["weighted_ecl"]) for _, row in stage.iterrows()}

    context_bar(
        [
            ("As at", au_date("2025-03-01")),
            ("Run", "ecl_v1"),
            ("Portfolio", "Freddie Mac Case Study"),
            ("Currency", "USD"),
        ]
    )
    executive_kpis(
        [
            ("Total EAD", usd(kpis["total_ead"]), "Exposure used for ECL"),
            ("Weighted ECL", usd(kpis["weighted_ecl"]), "Probability-weighted total"),
            ("Coverage", percentage(kpis["coverage_ratio"]), "Weighted ECL / EAD"),
        ],
        columns=3,
    )
    executive_kpis(
        [
            ("Stage 1 ECL", usd(stage_ecl[1]), "12-month ECL"),
            ("Stage 2 ECL", usd(stage_ecl[2]), "Lifetime ECL"),
            ("Stage 3 ECL", usd(stage_ecl[3]), "Credit-impaired approximation"),
        ],
        columns=3,
    )

    section_header("Stage contribution", "Weighted ECL amount and share by IFRS 9 stage.")
    stage_chart = stage.copy()
    stage_chart["Stage"] = stage_chart["stage"].map(lambda value: f"Stage {int(value)}")
    st.plotly_chart(
        horizontal_contribution_bar(
            stage_chart,
            label="Stage",
            value="weighted_ecl",
            title="Weighted ECL by stage",
            xaxis_title="Weighted ECL",
        ),
        width="stretch",
    )
    source_caption("As at 1 Mar 2025 · Source: ecl_v1")

    left, right = st.columns(2)
    with left:
        section_header("Scenario variance", "ECL movement relative to Base scenario.")
        st.plotly_chart(
            scenario_delta_bar(scenario, title="Scenario ECL delta versus Base"),
            width="stretch",
        )
    with right:
        section_header("Rating concentration", "Ranked ECL contribution by rating.")
        ranked = rating.sort_values("weighted_ecl", ascending=False).copy()
        st.plotly_chart(
            horizontal_contribution_bar(
                ranked,
                label="rating",
                value="weighted_ecl",
                title="Weighted ECL by rating",
                xaxis_title="Weighted ECL",
            ),
            width="stretch",
        )

    section_header("Structural versus downturn LGD sensitivity")
    comparison = _downturn_comparison(downturn)
    delta = comparison["Downturn"].iloc[0] - comparison["Structural"].iloc[0]
    structural_ecl = comparison["Structural"].iloc[0]
    delta_pct = delta / structural_ecl if structural_ecl else 0.0
    st.plotly_chart(
        dumbbell_chart(
            comparison,
            category="Metric",
            left_value="Structural",
            right_value="Downturn",
            left_label="Structural LGD",
            right_label="Downturn LGD sensitivity",
            title="Structural base versus downturn sensitivity",
            xaxis_title="Weighted ECL",
            formatter=usd,
        ),
        width="stretch",
    )
    st.caption(
        f"Downturn sensitivity: {signed_usd(delta)} ({percentage(delta_pct)}) "
        "versus structural base."
    )

    section_header("Stage analytical table", "Stage, EAD, ECL, coverage and concentration shares.")
    st.dataframe(format_table(stage_concentration_table(stage)), width="stretch", hide_index=True)

    methodology_note(
        "Baseline LGD is structural and macro-neutral. Stage 3 is approximated as current exposure "
        "times LGD. EAD uses contractual amortisation as the selected baseline. Quarterly macro "
        "anchors label PD scenario paths; monthly ECL horizons start at month 1 after reporting "
        "date."
    )
    render_guidance("ecl")
    with st.expander("Technical diagnostics", expanded=False):
        st.dataframe(format_table(scenario), width="stretch", hide_index=True)
        st.dataframe(format_table(downturn), width="stretch", hide_index=True)
        st.dataframe(format_table(artifacts["warnings"]), width="stretch", hide_index=True)

    if is_cloud_demo():
        st.info(full_mode_message())
    elif st.button("Run ECL", type="primary"):
        try:
            with st.spinner("Running ECL backend..."):
                result = run_ecl_from_ui()
        except Exception as exc:
            friendly_error(exc)
        else:
            st.success(f"ECL run completed: `{result.run_id}`")


main()
