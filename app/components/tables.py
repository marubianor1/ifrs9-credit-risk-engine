"""Reusable table styling helpers."""

from __future__ import annotations

import pandas as pd

from app.components.formatting import SPLIT_ORDER, bps, count, display_label, percentage, ratio, usd
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
                "Split": row["split"],
                "Observed default rate": observed,
                "Predicted PD": predicted,
                "Difference": diff,
                "O/E": oe,
                "Calibration status": status_label(calibration_status(oe)),
            }
        )
    output = pd.DataFrame(rows)
    output["Split"] = pd.Categorical(output["Split"], categories=SPLIT_ORDER, ordered=True)
    return output.sort_values("Split").reset_index(drop=True)


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
    display = frame.rename(columns={column: display_label(column) for column in frame.columns})
    styler = display.style.set_properties(
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
    numeric_cols = display.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        styler = styler.set_properties(subset=numeric_cols, **{"text-align": "right"})
    if "O/E" in display.columns:
        styler = styler.applymap(_oe_cell_style, subset=["O/E"])
    if "Calibration status" in display.columns:
        styler = styler.applymap(_status_cell_style, subset=["Calibration status"])
    return styler.format(_formatters(display))


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
        elif "difference" in name:
            formatters[column] = bps
        elif any(token in name for token in ["rate", "coverage", "pd", "psi", "auc", "gini", "ks"]):
            formatters[column] = percentage
        elif name == "o/e":
            formatters[column] = ratio
        elif any(token in name for token in ["loans", "rows", "defaults", "population"]):
            formatters[column] = count
    return formatters


def _oe_cell_style(value: float) -> str:
    status = calibration_status(float(value))
    return _status_fill(status)


def _status_cell_style(value: str) -> str:
    return _status_fill(_status_key(value))


def _status_key(value: object) -> str:
    text = str(value).upper()
    for status in ("GREEN", "AMBER", "RED"):
        if status in text:
            return status
    return text


def _status_fill(status: str) -> str:
    colour = STATUS_COLOURS.get(status)
    if not colour:
        return ""
    fill = {
        POSITIVE: "#E8F3EE",
        WARNING: "#FFF4DE",
        ADVERSE: "#FDECEC",
    }.get(colour, "#FFFFFF")
    return f"background-color: {fill}; color: #1D2939"
