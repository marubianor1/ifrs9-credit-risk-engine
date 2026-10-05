"""Shared visual theme for the IFRS 9 Streamlit app."""

from __future__ import annotations

import plotly.io as pio
import streamlit as st

from app.components.formatting import APP_VERSION

NAVY = "#17324D"
BLUE = "#286090"
TEAL = "#287C8E"
POSITIVE = "#2E7D5B"
WARNING = "#B7791F"
ADVERSE = "#B42318"
TEXT = "#1D2939"
SECONDARY_TEXT = "#667085"
GRID = "#E4E7EC"
BACKGROUND = "#F8FAFC"
SURFACE = "#FFFFFF"

STAGE_COLOURS = {1: BLUE, 2: WARNING, 3: ADVERSE}
SCENARIO_COLOURS = {"BASE": BLUE, "UPSIDE": TEAL, "DOWNSIDE": ADVERSE, "WEIGHTED": NAVY}
RATING_COLOURS = {
    "R1": "#2E7D5B",
    "R2": "#5D9C72",
    "R3": "#B7791F",
    "R4": "#C65D2E",
    "R5": "#B42318",
}
STATUS_COLOURS = {"GREEN": POSITIVE, "AMBER": WARNING, "RED": ADVERSE}
STATUS_ICONS = {"GREEN": "OK", "AMBER": "WATCH", "RED": "ACTION"}


def stage_colour(stage: int | str) -> str:
    """Return the semantic stage colour."""
    return STAGE_COLOURS[int(stage)]


def scenario_colour(scenario: str) -> str:
    """Return the semantic scenario colour."""
    return SCENARIO_COLOURS[str(scenario).upper()]


def rating_order() -> list[str]:
    """Return rating grades from lower to higher credit risk."""
    return ["R1", "R2", "R3", "R4", "R5"]


def rating_colour(rating: str) -> str:
    """Return rating colour progressing from lower to higher risk."""
    return RATING_COLOURS[str(rating)]


def status_colour(status: str) -> str:
    """Return status colour."""
    return STATUS_COLOURS[str(status).upper()]


def status_label(status: str) -> str:
    """Return text-plus-marker status label."""
    key = str(status).upper()
    return f"{STATUS_ICONS[key]} - {key}"


def apply_plotly_template() -> None:
    """Register and apply the global Plotly template."""
    pio.templates["ifrs9_v2"] = {
        "layout": {
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "font": {"color": TEXT, "family": "Arial, sans-serif", "size": 13},
            "title": {"font": {"color": TEXT, "size": 18}},
            "colorway": [BLUE, TEAL, POSITIVE, WARNING, ADVERSE, NAVY],
            "xaxis": {
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "title": {"font": {"color": SECONDARY_TEXT}},
            },
            "yaxis": {
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "title": {"font": {"color": SECONDARY_TEXT}},
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            "margin": {"l": 40, "r": 20, "t": 58, "b": 48},
        }
    }
    pio.templates.default = "ifrs9_v2"


def apply_streamlit_theme() -> None:
    """Apply lightweight V2 styling to Streamlit surfaces."""
    apply_plotly_template()
    st.markdown(
        f"""
        <style>
        :root {{
            --ifrs9-navy: {NAVY};
            --ifrs9-blue: {BLUE};
            --ifrs9-text: {TEXT};
            --ifrs9-muted: {SECONDARY_TEXT};
            --ifrs9-grid: {GRID};
            --ifrs9-bg: {BACKGROUND};
        }}
        .stApp {{
            background: {BACKGROUND};
            color: {TEXT};
        }}
        h1, h2, h3 {{
            color: {TEXT};
            letter-spacing: 0;
        }}
        [data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {GRID};
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {SECONDARY_TEXT};
        }}
        [data-testid="stSidebar"] {{
            background: #FFFFFF;
        }}
        .ifrs9-version {{
            color: {SECONDARY_TEXT};
            font-size: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def version_caption() -> str:
    """Return the V2 app version caption."""
    return f"App version {APP_VERSION}"
