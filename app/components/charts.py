"""Shared Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.formatting import display_label, percentage, signed_usd, usd
from app.components.theme import (
    GRID,
    RATING_COLOURS,
    SCENARIO_COLOURS,
    STAGE_COLOURS,
    TEXT,
    apply_plotly_template,
    rating_order,
)


def finish_chart(fig: go.Figure, *, yaxis_title: str = "", xaxis_title: str = "") -> go.Figure:
    """Apply shared accessibility and layout standards."""
    apply_plotly_template()
    fig.update_layout(
        template="ifrs9_v2",
        autosize=True,
        hovermode="x unified",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin={"l": 36, "r": 18, "t": 44, "b": 40},
        modebar_remove=[
            "select2d",
            "lasso2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
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
    data["IFRS 9 Stage"] = data["stage"].map(lambda value_: f"Stage {int(value_)}")
    colours = {f"Stage {stage}": colour for stage, colour in STAGE_COLOURS.items()}
    is_rate = "coverage" in value
    fig = px.bar(
        data,
        x="IFRS 9 Stage",
        y=value,
        color="IFRS 9 Stage",
        color_discrete_map=colours,
        title=title,
        labels={value: yaxis_title, "IFRS 9 Stage": "IFRS 9 Stage"},
        text=data[value].map(percentage if is_rate else usd),
    )
    hover = "%{x}<br>%{y:.2%}<extra></extra>" if is_rate else "%{x}<br>%{y:,.2f}<extra></extra>"
    fig.update_traces(textposition="outside", hovertemplate=hover, showlegend=False)
    if is_rate:
        fig.update_yaxes(tickformat=".0%")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="")


def stage_mix_bar(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Build a 100 percent horizontal stacked exposure mix chart."""
    data = frame.copy()
    total = float(data["total_ead"].sum())
    data["share"] = data["total_ead"] / total if total else 0.0
    data["IFRS 9 Stage"] = data["stage"].map(lambda value: f"Stage {int(value)}")
    fig = go.Figure()
    for _, row in data.iterrows():
        stage_name = row["IFRS 9 Stage"]
        fig.add_trace(
            go.Bar(
                y=["Portfolio EAD"],
                x=[row["share"]],
                name=stage_name,
                orientation="h",
                marker_color=STAGE_COLOURS[int(row["stage"])],
                text=[percentage(row["share"])],
                textposition="inside",
                customdata=[[usd(row["total_ead"])]],
                hovertemplate=(
                    f"{stage_name}<br>Share: %{{x:.2%}}<br>"
                    "EAD: %{customdata[0]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(title=title, barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return finish_chart(fig, yaxis_title="", xaxis_title="Share of EAD")


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
        labels={"scenario": "Scenario", "ecl": "ECL"},
        text=data["ecl"].map(usd),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        showlegend=False,
    )
    return finish_chart(fig, yaxis_title="ECL", xaxis_title="")


def scenario_delta_bar(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Build Delta ECL versus Base scenario chart."""
    data = frame.copy()
    data["Scenario"] = data["scenario"].astype(str).str.upper()
    base = float(data.loc[data["Scenario"].eq("BASE"), "ecl"].iloc[0])
    order = ["UPSIDE", "BASE", "DOWNSIDE"]
    data = data[data["Scenario"].isin(order)].copy()
    data["delta_ecl"] = data["ecl"] - base
    data["delta_pct"] = data["delta_ecl"] / base if base else 0.0
    fig = px.bar(
        data,
        x="Scenario",
        y="delta_ecl",
        color="Scenario",
        category_orders={"Scenario": order},
        color_discrete_map=SCENARIO_COLOURS,
        title=title,
        labels={"delta_ecl": "Delta ECL vs Base", "Scenario": "Scenario"},
        text=data["delta_ecl"].map(signed_usd),
    )
    fig.add_hline(y=0, line_color="#667085", line_width=1)
    fig.update_traces(
        textposition="outside",
        customdata=data[["delta_pct"]],
        hovertemplate="%{x}<br>Delta: %{y:,.2f}<br>Delta %: %{customdata[0]:.2%}<extra></extra>",
        showlegend=False,
    )
    return finish_chart(fig, yaxis_title="Delta ECL vs Base", xaxis_title="")


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
        labels={"rating": "Rating", value: yaxis_title},
        text=data[value].map(usd if "ecl" in value or "ead" in value else percentage),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        showlegend=False,
    )
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
        labels={x: display_label(x), y: yaxis_title, color or "": display_label(color or "")},
        text=frame[y].map(lambda value: f"{value:,.2f}"),
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,.4f}<extra></extra>")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="")


def calibration_scatter(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Build a credit-risk calibration scatter with observed equals predicted reference."""
    data = frame.copy()
    if "split" in data.columns:
        data["split"] = pd.Categorical(
            data["split"],
            categories=["TRAIN", "VALIDATION", "OOT"],
            ordered=True,
        )
        data = data.sort_values("split")
    fig = px.scatter(
        data,
        x="predicted_bad_rate",
        y="observed_bad_rate",
        color="split" if "split" in data.columns else None,
        title=title,
        labels={
            "predicted_bad_rate": "Predicted PD",
            "observed_bad_rate": "Observed default rate",
            "split": "Split",
        },
        hover_data={
            "predicted_bad_rate": ":.2%",
            "observed_bad_rate": ":.2%",
            "split": True,
        },
    )
    max_value = max(float(data["predicted_bad_rate"].max()), float(data["observed_bad_rate"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="Observed = Predicted",
            line={"color": "#667085", "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 1, "color": "white"}})
    fig.update_xaxes(tickformat=".1%", range=[0, max_value * 1.08])
    fig.update_yaxes(tickformat=".1%", range=[0, max_value * 1.08])
    return finish_chart(fig, yaxis_title="Observed default rate", xaxis_title="Predicted PD")
