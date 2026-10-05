"""Display formatting helpers for the Streamlit app."""

from __future__ import annotations

from datetime import date, datetime

APP_VERSION = "2.0.0-beta"


def usd(value: float, *, decimals: int = 1) -> str:
    """Format Freddie Mac case-study currency as USD."""
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"US${float(value) / 1_000_000_000:,.{decimals}f}bn"
    if abs_value >= 1_000_000:
        return f"US${float(value) / 1_000_000:,.{decimals}f}m"
    return f"US${float(value):,.0f}"


def percentage(value: float, *, decimals: int = 2) -> str:
    """Format a decimal risk rate as a percentage."""
    return f"{float(value) * 100:,.{decimals}f}%"


def bps(value: float, *, decimals: int = 0) -> str:
    """Format a decimal percentage-point difference as basis points."""
    return f"{float(value) * 10_000:,.{decimals}f} bps"


def count(value: float | int) -> str:
    """Format a count with thousands separators."""
    return f"{int(round(float(value))):,}"


def ratio(value: float, *, decimals: int = 2) -> str:
    """Format a unitless ratio such as O/E."""
    return f"{float(value):,.{decimals}f}"


def au_date(value: str | date | datetime) -> str:
    """Format dates using Australian presentation conventions."""
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value).date()
    elif isinstance(value, datetime):
        parsed = value.date()
    else:
        parsed = value
    return f"{parsed.day} {parsed.strftime('%b %Y')}"


def format_by_name(column: str, value: object) -> str:
    """Apply a sensible table formatter from a column name."""
    if value is None:
        return ""
    name = column.lower()
    if any(token in name for token in ["ead", "ecl", "amount"]):
        return usd(float(value))
    if any(token in name for token in ["coverage", "bad_rate", "pd", "psi", "gini", "auc"]):
        return percentage(float(value))
    if "oe_ratio" in name:
        return ratio(float(value))
    if any(token in name for token in ["loans", "rows", "defaults", "population"]):
        return count(float(value))
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)
