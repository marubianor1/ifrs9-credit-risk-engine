"""Shared Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.formatting import SPLIT_ORDER, display_label, percentage, signed_usd, usd
from app.components.theme import (
    GRID,
    RATING_COLOURS,
    SCENARIO_COLOURS,
    SECONDARY_TEXT,
    STAGE_COLOURS,
    TEXT,
    apply_plotly_template,
    rating_order,
)

__all__ = [
    "calibration_scatter",
    "dumbbell_chart",
    "finish_chart",
    "horizontal_contribution_bar",
    "metric_bar",
    "oe_bullet_chart",
    "rating_bar",
    "scenario_waterfall",
    "split_grouped_bar",
    "scenario_bar",
    "scenario_delta_bar",
    "stage_bar",
    "stage_concentration_chart",
    "stage_mix_bar",
]

NEUTRAL_SPLIT_COLOURS = {
    "TRAIN": "#286090",
    "VALIDATION": "#667085",
    "OOT": "#98A2B3",
}


def finish_chart(fig: go.Figure, *, yaxis_title: str = "", xaxis_title: str = "") -> go.Figure:
    """Apply shared accessibility and layout standards."""
    apply_plotly_template()
    fig.update_layout(
        template="ifrs9_v3",
        autosize=True,
        hovermode="x unified",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin={"l": 36, "r": 18, "t": 36, "b": 36},
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


def horizontal_contribution_bar(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    xaxis_title: str,
    colour: str = "#2457A6",
    formatter=usd,
) -> go.Figure:
    """Build a ranked horizontal contribution chart."""
    data = frame.copy().sort_values(value, ascending=True)
    fig = go.Figure(
        go.Bar(
            y=data[label].astype(str),
            x=data[value],
            orientation="h",
            marker_color=colour,
            text=data[value].map(formatter),
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, showlegend=False)
    return finish_chart(fig, yaxis_title="", xaxis_title=xaxis_title)


def dumbbell_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    left_value: str,
    right_value: str,
    left_label: str,
    right_label: str,
    title: str,
    xaxis_title: str,
    formatter=percentage,
) -> go.Figure:
    """Build a connected-dot comparison chart."""
    data = frame.copy()
    y_values = data[category].astype(str).tolist()
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[left_value], row[right_value]],
                y=[str(row[category]), str(row[category])],
                mode="lines",
                line={"color": "#D0D5DD", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data[left_value],
            y=y_values,
            mode="markers+text",
            name=left_label,
            marker={"color": "#5B7FA3", "size": 10, "line": {"color": "#FFFFFF", "width": 1}},
            text=data[left_value].map(formatter),
            textposition="middle left",
            hovertemplate=f"{left_label}: %{{x:,.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[right_value],
            y=y_values,
            mode="markers+text",
            name=right_label,
            marker={"color": "#2457A6", "size": 10, "line": {"color": "#FFFFFF", "width": 1}},
            text=data[right_value].map(formatter),
            textposition="middle right",
            hovertemplate=f"{right_label}: %{{x:,.4f}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, legend={"orientation": "h", "y": 1.04})
    return finish_chart(fig, yaxis_title="", xaxis_title=xaxis_title)


def oe_bullet_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    oe_value: str,
    title: str,
    thresholds: dict | None = None,
) -> go.Figure:
    """Build O/E bullet visuals centred on 1.00."""
    data = frame.copy()
    y_values = data[category].astype(str)
    fig = go.Figure()
    if thresholds:
        bands = [
            (thresholds.get("amber_min", 0.0), thresholds.get("green_min", 0.8), "#FDECEC"),
            (thresholds.get("green_min", 0.8), thresholds.get("green_max", 1.2), "#E8F3EE"),
            (thresholds.get("green_max", 1.2), thresholds.get("amber_max", 1.5), "#FFF4DE"),
        ]
        for low, high, colour in bands:
            fig.add_shape(
                type="rect",
                x0=low,
                x1=high,
                y0=-0.5,
                y1=len(data) - 0.5,
                fillcolor=colour,
                opacity=0.55,
                line_width=0,
                layer="below",
            )
    fig.add_vline(x=1.0, line_color=SECONDARY_TEXT, line_dash="dash", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=data[oe_value],
            y=y_values,
            mode="markers+text",
            marker={"color": "#2457A6", "size": 11},
            text=data[oe_value].map(lambda value: f"{float(value):.2f}"),
            textposition="middle right",
            hovertemplate="%{y}<br>O/E: %{x:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(range=[0, max(1.6, float(data[oe_value].max()) * 1.15)])
    return finish_chart(fig, yaxis_title="", xaxis_title="Observed / expected")


def stage_concentration_chart(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Compare EAD share and ECL share by IFRS 9 stage."""
    data = frame.copy()
    total_ead = float(data["total_ead"].sum())
    total_ecl = float(data["weighted_ecl"].sum())
    data["EAD share"] = data["total_ead"] / total_ead if total_ead else 0.0
    data["ECL share"] = data["weighted_ecl"] / total_ecl if total_ecl else 0.0
    data["Stage"] = data["stage"].map(lambda value: f"Stage {int(value)}")
    chart = data.melt(
        id_vars=["Stage", "stage"],
        value_vars=["EAD share", "ECL share"],
        var_name="Measure",
        value_name="Share",
    )
    fig = go.Figure()
    for measure, colour in [("EAD share", "#5B7FA3"), ("ECL share", "#2457A6")]:
        subset = chart[chart["Measure"].eq(measure)]
        fig.add_trace(
            go.Bar(
                y=subset["Stage"],
                x=subset["Share"],
                orientation="h",
                name=measure,
                marker_color=colour,
                text=subset["Share"].map(lambda value: percentage(value, decimals=1)),
                textposition="outside",
                hovertemplate=f"{measure}<br>%{{y}}: %{{x:.2%}}<extra></extra>",
            )
        )
    fig.update_layout(title=title, barmode="group")
    fig.update_xaxes(tickformat=".0%")
    return finish_chart(fig, yaxis_title="", xaxis_title="Share of portfolio")


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


def split_grouped_bar(
    frame: pd.DataFrame,
    *,
    y: str | list[str],
    title: str,
    yaxis_title: str,
    rate_axis: bool = False,
) -> go.Figure:
    """Build a grouped split chart in TRAIN, VALIDATION, OOT order."""
    data = frame.copy()
    if "split" in data.columns:
        data["split"] = pd.Categorical(data["split"], categories=SPLIT_ORDER, ordered=True)
        data = data.sort_values("split")
    labels = {"split": "Split"}
    if isinstance(y, list):
        labels.update({column: display_label(column) for column in y})
    else:
        labels[y] = yaxis_title
    fig = px.bar(
        data,
        x="split",
        y=y,
        barmode="group",
        title=title,
        labels=labels,
        color_discrete_sequence=["#286090", "#287C8E", "#B7791F", "#667085"],
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>")
    if rate_axis:
        fig.update_yaxes(tickformat=".1%")
        fig.update_traces(hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
    return finish_chart(fig, yaxis_title=yaxis_title, xaxis_title="")


def scenario_waterfall(frame: pd.DataFrame, *, title: str) -> go.Figure:
    """Build an ECL delta waterfall from scenario driver effects."""
    data = frame.copy()
    total = float(data["effect"].sum())
    labels = data["driver"].astype(str).tolist() + ["Total delta"]
    values = data["effect"].astype(float).tolist() + [total]
    measures = ["relative"] * len(data) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            text=[signed_usd(value) for value in values],
            textposition="outside",
            connector={"line": {"color": "#D0D5DD"}},
            increasing={"marker": {"color": "#B42318"}},
            decreasing={"marker": {"color": "#287C8E"}},
            totals={"marker": {"color": "#667085"}},
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, showlegend=False)
    return finish_chart(fig, yaxis_title="Delta ECL", xaxis_title="")


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
