"""Multipage Streamlit entrypoint for the IFRS 9 portfolio application."""

from __future__ import annotations

import streamlit as st

from app.components.layout import render_sidebar
from app.state import init_state

st.set_page_config(
    page_title="IFRS 9 Credit Risk Engine",
    page_icon="",
    layout="wide",
)

init_state()
render_sidebar()

pages = [
    st.Page("pages/overview.py", title="Overview"),
    st.Page("pages/scoring.py", title="Scoring Models"),
    st.Page("pages/pd.py", title="PD"),
    st.Page("pages/scenario_lab.py", title="Scenario Lab"),
]

navigation = st.navigation(pages)
navigation.run()
