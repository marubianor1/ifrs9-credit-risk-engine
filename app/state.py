"""Session-state defaults for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def init_state() -> None:
    """Initialize app session state."""
    defaults = {
        "portfolio_mode": "Demo / Existing Runs",
        "scorecard_run": "behavioural_qe_v1",
        "pd_run": "pd_behavioural_qe_v1",
        "ecl_run": "ecl_v1",
        "scenario_preset": "baseline",
        "last_scenario_run_id": None,
        "last_scenario_cached": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
