"""Multipage Streamlit entrypoint for the IFRS 9 portfolio application."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.layout import render_sidebar  # noqa: E402
from app.components.theme import apply_streamlit_theme  # noqa: E402
from app.services.artifacts import configure_mode_from_streamlit_secrets  # noqa: E402
from app.state import init_state  # noqa: E402

st.set_page_config(
    page_title="IFRS 9 Credit Risk Engine",
    page_icon="",
    layout="wide",
)

apply_streamlit_theme()
configure_mode_from_streamlit_secrets(st.secrets)
init_state()
render_sidebar()

pages = {
    "Portfolio": [
        st.Page("pages/overview.py", title="Overview"),
        st.Page("pages/ecl.py", title="ECL"),
        st.Page("pages/scenario_lab.py", title="Scenario Lab"),
    ],
    "Risk Models": [
        st.Page("pages/scoring.py", title="Scoring Models"),
        st.Page("pages/pd.py", title="PD"),
        st.Page("pages/lgd.py", title="LGD"),
        st.Page("pages/ead.py", title="EAD"),
        st.Page("pages/sicr.py", title="SICR & Staging"),
    ],
    "Governance": [
        st.Page("pages/monitoring.py", title="Model Monitoring"),
        st.Page("pages/report_generator.py", title="Report Generator"),
        st.Page("pages/about.py", title="Methodology"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
