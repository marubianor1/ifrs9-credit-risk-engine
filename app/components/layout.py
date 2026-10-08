"""Shared layout helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from app.components.formatting import au_date, percentage, usd
from app.components.theme import version_caption
from app.services.artifacts import data_period, mode_label


def money(value: float) -> str:
    """Format a currency value compactly."""
    return usd(value)


def pct(value: float) -> str:
    """Format a decimal percentage."""
    return percentage(value)


def page_title(title: str, caption: str) -> None:
    """Render a consistent page title."""
    st.title(title)
    st.caption(caption)


def render_sidebar() -> None:
    """Render global project metadata in the sidebar."""
    with st.sidebar:
        st.markdown("### IFRS 9 Credit Risk Engine")
        st.caption("Portfolio analytics & model governance")
        st.selectbox(
            "Portfolio mode",
            ["Demo / Existing Runs"],
            key="portfolio_mode",
            help="The app starts from persisted artifacts and does not retrain on load.",
        )
        st.divider()
        st.caption(f"Execution mode: {mode_label()}")
        st.caption(f"Reporting date: {au_date('2025-03-01')}")
        st.caption(f"Data period: {data_period()}")
        st.divider()
        st.caption("Freddie Mac case study")
        st.caption("App V3 beta")
        st.caption(version_caption())
        st.caption("Portfolio/research implementation, not regulatory production readiness.")


def friendly_error(error: Exception) -> None:
    """Render a friendly error without exposing a stack trace."""
    st.error(str(error))
    st.caption("Check that the expected artifact exists, or run the relevant backend command.")
