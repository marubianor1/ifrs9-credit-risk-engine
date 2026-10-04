"""Shared layout helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from app.services.artifacts import data_period, git_commit, mode_label


def money(value: float) -> str:
    """Format a currency value compactly."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    """Format a decimal percentage."""
    return f"{value * 100:,.2f}%"


def page_title(title: str, caption: str) -> None:
    """Render a consistent page title."""
    st.title(title)
    st.caption(caption)


def render_sidebar() -> None:
    """Render global project metadata in the sidebar."""
    with st.sidebar:
        st.header("IFRS 9 Portfolio")
        st.selectbox(
            "Portfolio mode",
            ["Demo / Existing Runs"],
            key="portfolio_mode",
            help="Phase 1 starts from persisted artifacts and does not retrain on load.",
        )
        st.divider()
        st.metric("Project status", "Phase 3 UI")
        st.metric("Execution mode", mode_label())
        st.caption("Portfolio/research implementation, not regulatory production readiness.")
        st.write("Reporting date: `2025-03-01`")
        st.write(f"Git commit: `{git_commit()}`")
        st.write(f"Data period: {data_period()}")
        st.divider()
        st.write("Selected runs")
        st.write(f"Scorecard: `{st.session_state.get('scorecard_run', 'behavioural_qe_v1')}`")
        st.write(f"PD: `{st.session_state.get('pd_run', 'pd_behavioural_qe_v1')}`")
        st.write("LGD: `lgd_v1_2`")
        st.write("EAD: `ead_v1`")
        st.write("Staging: `sicr_v1_1`")
        st.write(f"ECL: `{st.session_state.get('ecl_run', 'ecl_v1')}`")


def friendly_error(error: Exception) -> None:
    """Render a friendly error without exposing a stack trace."""
    st.error(str(error))
    st.caption("Check that the expected artifact exists, or run the relevant backend command.")
