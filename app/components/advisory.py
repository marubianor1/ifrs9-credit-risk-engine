"""Advisory-grade presentation components for V3 pages."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st

from app.components.formatting import percentage, signed_usd, usd
from app.components.theme import (
    ADVERSE,
    BLUE,
    BORDER,
    SECONDARY_TEXT,
    SURFACE,
    TEXT,
    WARNING,
)


@dataclass(frozen=True)
class ExecutiveInsight:
    """A deterministic insight derived from validated aggregate artifacts."""

    text: str
    source: str


def page_header(title: str, purpose: str) -> None:
    """Render a restrained analytical page header."""
    st.markdown(
        f"""
        <div class="ifrs9-page-header">
          <div class="ifrs9-page-title">{escape(title)}</div>
          <div class="ifrs9-page-purpose">{escape(purpose)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_bar(items: list[tuple[str, str]]) -> None:
    """Render compact run/reporting-date context."""
    parts = [
        f"<span class='ifrs9-context-label'>{escape(label)}</span> {escape(value)}"
        for label, value in items
    ]
    st.markdown(
        f"<div class='ifrs9-context-bar'>{' <span>•</span> '.join(parts)}</div>",
        unsafe_allow_html=True,
    )


def executive_kpis(items: list[tuple[str, str, str | None]], *, columns: int = 3) -> None:
    """Render professional KPI tiles with value-first hierarchy."""
    if not items:
        return
    for start in range(0, len(items), columns):
        row = items[start : start + columns]
        cols = st.columns(len(row))
        for col, (label, value, comparison) in zip(cols, row, strict=True):
            comparison_html = (
                f"<div class='ifrs9-kpi-comparison'>{escape(comparison)}</div>"
                if comparison
                else ""
            )
            col.markdown(
                f"""
                <div class="ifrs9-kpi">
                  <div class="ifrs9-kpi-value">{escape(value)}</div>
                  <div class="ifrs9-kpi-label">{escape(label)}</div>
                  {comparison_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def insight_callout(insight: ExecutiveInsight | str, *, source: str | None = None) -> None:
    """Render a subtle executive insight callout."""
    text = insight.text if isinstance(insight, ExecutiveInsight) else insight
    source_text = insight.source if isinstance(insight, ExecutiveInsight) else source
    source_html = (
        f"<div class='ifrs9-insight-source'>{escape(source_text)}</div>" if source_text else ""
    )
    st.markdown(
        f"""
        <div class="ifrs9-insight">
          <div class="ifrs9-insight-title">Executive insight</div>
          <div class="ifrs9-insight-text">{escape(text)}</div>
          {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a compact section header."""
    subtitle_html = (
        f"<div class='ifrs9-section-subtitle'>{escape(subtitle)}</div>" if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="ifrs9-section-header">
          <div class="ifrs9-section-title">{escape(title)}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_container(title: str, subtitle: str | None = None, *, source: str | None = None):
    """Open a chart section with title and optional source caption."""
    section_header(title, subtitle)
    if source:
        source_caption(source)


def source_caption(text: str) -> None:
    """Render source text under analytical content."""
    st.caption(text)


def methodology_note(text: str, *, expanded: bool = False) -> None:
    """Render visually secondary methodology guidance."""
    with st.expander("Methodology & interpretation", expanded=expanded):
        st.write(text)


def status_badge(status: str) -> str:
    """Return an HTML status badge with text plus colour."""
    key = str(status).upper()
    colour = {"GREEN": "#2E7D5B", "AMBER": WARNING, "RED": ADVERSE}.get(key, SECONDARY_TEXT)
    return (
        f"<span class='ifrs9-status-badge' style='border-color:{colour};"
        f"color:{colour};'>{escape(key)}</span>"
    )


def analytical_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return a consulting-grade table style."""
    styler = frame.style.set_properties(
        **{
            "border-bottom": f"1px solid {BORDER}",
            "color": TEXT,
            "font-size": "13px",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("border-bottom", f"1px solid {BORDER}"),
                    ("color", SECONDARY_TEXT),
                    ("font-weight", "600"),
                    ("font-size", "12px"),
                    ("text-align", "left"),
                    ("background", SURFACE),
                ],
            },
            {"selector": "td", "props": [("padding", "0.42rem 0.55rem")]},
            {"selector": "tbody tr:last-child td", "props": [("border-bottom", "0")]},
        ]
    )
    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        styler = styler.set_properties(subset=numeric_cols, **{"text-align": "right"})
    return styler


def stage_concentration_table(stage: pd.DataFrame) -> pd.DataFrame:
    """Build Stage table with EAD/ECL shares from validated stage aggregates."""
    output = stage.copy()
    total_ead = float(output["total_ead"].sum())
    total_ecl = float(output["weighted_ecl"].sum())
    output["% EAD"] = output["total_ead"] / total_ead if total_ead else 0.0
    output["% ECL"] = output["weighted_ecl"] / total_ecl if total_ecl else 0.0
    return pd.DataFrame(
        {
            "Stage": output["stage"].map(lambda value: f"Stage {int(value)}"),
            "EAD": output["total_ead"].map(usd),
            "Weighted ECL": output["weighted_ecl"].map(usd),
            "Coverage": output["coverage_ratio"].map(percentage),
            "% EAD": output["% EAD"].map(percentage),
            "% ECL": output["% ECL"].map(percentage),
        }
    )


def stage_concentration_insight(stage: pd.DataFrame, *, source: str) -> ExecutiveInsight | None:
    """Return Stage 2+3 exposure versus ECL concentration insight."""
    if not {"stage", "total_ead", "weighted_ecl"}.issubset(stage.columns):
        return None
    total_ead = float(stage["total_ead"].sum())
    total_ecl = float(stage["weighted_ecl"].sum())
    if not total_ead or not total_ecl:
        return None
    higher = stage.loc[stage["stage"].isin([2, 3])]
    exposure_share = float(higher["total_ead"].sum()) / total_ead
    ecl_share = float(higher["weighted_ecl"].sum()) / total_ecl
    return ExecutiveInsight(
        text=(
            "Stage 2 and Stage 3 represent "
            f"{percentage(exposure_share, decimals=1)} of portfolio exposure but "
            f"{percentage(ecl_share, decimals=1)} of expected credit loss."
        ),
        source=source,
    )


def rating_concentration_insight(rating: pd.DataFrame, *, source: str) -> ExecutiveInsight | None:
    """Return top-rating ECL contribution insight."""
    if not {"rating", "weighted_ecl"}.issubset(rating.columns):
        return None
    total = float(rating["weighted_ecl"].sum())
    if not total:
        return None
    top = rating.sort_values("weighted_ecl", ascending=False).iloc[0]
    share = float(top["weighted_ecl"]) / total
    return ExecutiveInsight(
        text=f"{top['rating']} contributes {percentage(share, decimals=1)} of portfolio ECL.",
        source=source,
    )


def stage_coverage_observation(stage: pd.DataFrame) -> str | None:
    """Return a directly supported Stage 3 coverage observation."""
    if "coverage_ratio" not in stage or "stage" not in stage:
        return None
    if not set(stage["stage"]).issuperset({1, 3}):
        return None
    stage_1 = float(stage.loc[stage["stage"].eq(1), "coverage_ratio"].iloc[0])
    stage_3 = float(stage.loc[stage["stage"].eq(3), "coverage_ratio"].iloc[0])
    return (
        f"Stage 3 coverage is {percentage(stage_3)}, compared with "
        f"{percentage(stage_1)} for Stage 1."
    )


def downside_observation(scenario: pd.DataFrame) -> str | None:
    """Return deterministic Downside versus Base observation."""
    data = scenario.copy()
    data["scenario"] = data["scenario"].astype(str).str.upper()
    if not {"BASE", "DOWNSIDE"}.issubset(set(data["scenario"])):
        return None
    base = float(data.loc[data["scenario"].eq("BASE"), "ecl"].iloc[0])
    downside = float(data.loc[data["scenario"].eq("DOWNSIDE"), "ecl"].iloc[0])
    delta = downside - base
    pct = delta / base if base else 0.0
    return f"Downside ECL is {signed_usd(delta)} ({percentage(pct)}) above Base."


def apply_advisory_css() -> None:
    """Inject restrained V3 component styling."""
    st.markdown(
        f"""
        <style>
        .ifrs9-page-header {{
            margin: 0 0 0.65rem 0;
        }}
        .ifrs9-page-title {{
            font-size: 2rem;
            line-height: 1.16;
            font-weight: 680;
            color: {TEXT};
        }}
        .ifrs9-page-purpose {{
            color: {SECONDARY_TEXT};
            font-size: 0.92rem;
            margin-top: 0.2rem;
        }}
        .ifrs9-context-bar {{
            border-top: 1px solid {BORDER};
            border-bottom: 1px solid {BORDER};
            padding: 0.48rem 0;
            margin: 0.7rem 0 1.2rem 0;
            color: {SECONDARY_TEXT};
            font-size: 0.78rem;
        }}
        .ifrs9-context-label {{
            color: {TEXT};
            font-weight: 600;
        }}
        .ifrs9-kpi {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            padding: 0.75rem 0.8rem;
            min-height: 5.1rem;
        }}
        .ifrs9-kpi-value {{
            color: {TEXT};
            font-weight: 690;
            font-size: 1.35rem;
            line-height: 1.16;
            white-space: nowrap;
        }}
        .ifrs9-kpi-label {{
            color: {SECONDARY_TEXT};
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }}
        .ifrs9-kpi-comparison {{
            color: {SECONDARY_TEXT};
            font-size: 0.72rem;
            margin-top: 0.2rem;
        }}
        .ifrs9-insight {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-left: 3px solid {BLUE};
            padding: 0.75rem 0.9rem;
            margin: 0.65rem 0 1.25rem 0;
        }}
        .ifrs9-insight-title {{
            color: {SECONDARY_TEXT};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.68rem;
            font-weight: 700;
        }}
        .ifrs9-insight-text {{
            color: {TEXT};
            font-size: 1rem;
            margin-top: 0.18rem;
        }}
        .ifrs9-insight-source {{
            color: {SECONDARY_TEXT};
            font-size: 0.72rem;
            margin-top: 0.2rem;
        }}
        .ifrs9-section-header {{
            margin: 1.65rem 0 0.45rem 0;
            padding-top: 0.3rem;
            border-top: 1px solid {BORDER};
        }}
        .ifrs9-section-title {{
            color: {TEXT};
            font-weight: 680;
            font-size: 1.05rem;
        }}
        .ifrs9-section-subtitle {{
            color: {SECONDARY_TEXT};
            font-size: 0.78rem;
            margin-top: 0.12rem;
        }}
        .ifrs9-status-badge {{
            display: inline-block;
            border: 1px solid;
            padding: 0.08rem 0.35rem;
            font-size: 0.7rem;
            font-weight: 700;
            background: #FFFFFF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
