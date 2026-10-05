"""Shared Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.formatting import percentage, usd
from app.components.theme import (
    GRID,
    RATING_COLOURS,
    SCENARIO_COLOURS,
    STAGE_COLOURS,
    TEXT,
    rating_order,
)


def finish_chart(fig: go.Figure, *, yaxis_title: str = "", xaxis_title: str = "") -> go.Figure:
    """Apply shared accessibility and layout standards."""
    fig.update_layout(
        template="ifrs9_v2",
        autosize=True,
        hovermode="x unified",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(showgrid=False, tickfont={"color": TEXT})
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero", tickfont={"color": TEXT})
    return fig


def stage_bar(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Build a semantic stage bar chart."""
    data = frame.copy()
    data["stage_label"] = data["stage"].map(lambda value_: f"Stage {int(value_)}")
    colours = {f"Stage {stage}": colour for stage, colour in STAGE_COLOURS.items()}
    fig = px.bar(
        data,
        x="stage_label",
        y=value,
        color="stage_label",
        color_discrete_map=colours,
        title=title,
        text=data[value].map(usd if "ecl" in value or "ead" in value else percentage),
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="")


def scenario_bar(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Build a Base/Upside/Downside scenario chart."""
    data = frame.copy()
    data["scenario"] = data["scenario"].astype(str).str.upper()
    order = ["UPSIDE", "BASE", "DOWNSIDE"]
    data = data[data["scenario"].isin(order)]
    fig = px.bar(
        data,
        x="scenario",
        y="ecl",
        color="scenario",
        category_orders={"scenario": order},
        color_discrete_map=SCENARIO_COLOURS,
        title=title,
        text=data["ecl"].map(usd),
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>")
    return finish_chart(fig, yaxis_title="ECL", xaxis_title="")


def rating_bar(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Build a rating chart with ordered credit-risk colours."""
    data = frame.copy()
    fig = px.bar(
        data,
        x="rating",
        y=value,
        color="rating",
        category_orders={"rating": rating_order()},
        color_discrete_map=RATING_COLOURS,
        title=title,
        text=data[value].map(usd if "ecl" in value or "ead" in value else percentage),
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="Rating")


def metric_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Build a generic professional bar chart."""
    fig = px.bar(
        frame,
        x=x,
        y=y,
        color=color,
        title=title,
        text=frame[y].map(lambda value: f"{value:,.2f}"),
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,.4f}<extra></extra>")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="")
