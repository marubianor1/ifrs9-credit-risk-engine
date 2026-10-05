"""Reusable table styling helpers."""

from __future__ import annotations

import pandas as pd

from app.components.formatting import bps, count, percentage, ratio, usd
from app.components.theme import ADVERSE, GRID, POSITIVE, STATUS_COLOURS, WARNING, status_label


def bad_rate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a professional bad-rate/calibration summary table."""
    required = {"split", "observed_bad_rate", "predicted_bad_rate"}
    if not required.issubset(metrics.columns):
        return metrics.copy()
    rows = []
    for _, row in metrics.iterrows():
        observed = float(row["observed_bad_rate"])
        predicted = float(row["predicted_bad_rate"])
        diff = observed - predicted
        oe = observed / predicted if predicted else None
        rows.append(
            {
                "split": row["split"],
                "observed_default_rate": observed,
                "predicted_pd": predicted,
                "difference_bps": diff,
                "oe_ratio": oe,
                "calibration_status": calibration_status(oe),
            }
        )
    return pd.DataFrame(rows)


def calibration_status(oe_ratio: float | None) -> str:
    """Classify O/E with transparent project-style monitoring bands."""
    if oe_ratio is None:
        return "UNAVAILABLE"
    if 0.8 <= oe_ratio <= 1.2:
        return "GREEN"
    if 0.6 <= oe_ratio <= 1.5:
        return "AMBER"
    return "RED"


def format_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return a consistently formatted dataframe Styler."""
    styler = frame.style.set_properties(
        **{
            "border-bottom": f"1px solid {GRID}",
            "text-align": "left",
            "color": "#1D2939",
        }
    ).set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("color", "#667085")]},
            {"selector": "td", "props": [("padding", "0.45rem 0.55rem")]},
        ]
    )
    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        styler = styler.set_properties(subset=numeric_cols, **{"text-align": "right"})
    if "calibration_status" in frame.columns:
        styler = styler.apply(_status_row_style, axis=1)
    return styler.format(_formatters(frame))


def add_status_label(frame: pd.DataFrame, column: str = "status") -> pd.DataFrame:
    """Add visible text marker for RAG statuses."""
    output = frame.copy()
    if column in output.columns:
        output[column] = output[column].map(status_label)
    return output


def _formatters(frame: pd.DataFrame) -> dict[str, object]:
    formatters: dict[str, object] = {}
    for column in frame.columns:
        name = column.lower()
        if any(token in name for token in ["ead", "ecl", "amount"]):
            formatters[column] = usd
        elif "difference_bps" in name:
            formatters[column] = bps
        elif any(token in name for token in ["rate", "coverage", "pd", "psi", "auc", "gini", "ks"]):
            formatters[column] = percentage
        elif "oe_ratio" in name:
            formatters[column] = ratio
        elif any(token in name for token in ["loans", "rows", "defaults", "population"]):
            formatters[column] = count
    return formatters


def _status_row_style(row: pd.Series) -> list[str]:
    status = str(row.get("calibration_status", "")).upper()
    colour = STATUS_COLOURS.get(status)
    if not colour:
        return ["" for _ in row]
    fill = {
        POSITIVE: "#E8F3EE",
        WARNING: "#FFF4DE",
        ADVERSE: "#FDECEC",
    }.get(colour, "#FFFFFF")
    return [f"background-color: {fill}; color: #1D2939" for _ in row]
