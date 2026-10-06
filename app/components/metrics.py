"""Reusable KPI components."""

from __future__ import annotations

import streamlit as st


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of KPI cards."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, value, help=help_text)
