"""Multipage Streamlit entrypoint for the IFRS 9 portfolio application."""

from __future__ import annotations

import streamlit as st

from app.components.layout import render_sidebar
from app.services.artifacts import configure_mode_from_streamlit_secrets
from app.state import init_state

st.set_page_config(
    page_title="IFRS 9 Credit Risk Engine",
    page_icon="",
    layout="wide",
)

configure_mode_from_streamlit_secrets(st.secrets)
init_state()
render_sidebar()

pages = [
    st.Page("pages/overview.py", title="Overview"),
    st.Page("pages/scoring.py", title="Scoring Models"),
    st.Page("pages/pd.py", title="PD"),
    st.Page("pages/lgd.py", title="LGD"),
    st.Page("pages/ead.py", title="EAD"),
    st.Page("pages/sicr.py", title="SICR & Staging"),
    st.Page("pages/ecl.py", title="ECL"),
    st.Page("pages/scenario_lab.py", title="Scenario Lab"),
    st.Page("pages/monitoring.py", title="Model Monitoring"),
    st.Page("pages/report_generator.py", title="Report Generator"),
    st.Page("pages/about.py", title="About / Methodology"),
]

navigation = st.navigation(pages)
navigation.run()
