"""Display formatting helpers for the Streamlit app."""

from __future__ import annotations

from datetime import date, datetime

APP_VERSION = "2.0.0-beta"
SPLIT_ORDER = ["TRAIN", "VALIDATION", "OOT"]

DISPLAY_LABELS = {
    "stage": "IFRS 9 Stage",
    "stage_label": "IFRS 9 Stage",
    "observed_bad_rate": "Observed default rate",
    "observed_default_rate": "Observed default rate",
    "predicted_bad_rate": "Predicted PD",
    "predicted_pd": "Predicted PD",
    "oe_ratio": "O/E",
    "difference_bps": "Difference",
    "scenario": "Scenario",
    "weighted_ecl": "Weighted ECL",
    "total_ead": "EAD",
    "coverage_ratio": "Coverage ratio",
    "roc_auc": "AUC",
    "gini": "Gini",
    "ks": "KS",
    "psi": "PSI",
    "split": "Split",
    "rating": "Rating",
}


def usd(value: float, *, decimals: int = 1) -> str:
    """Format Freddie Mac case-study currency as USD."""
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"US${float(value) / 1_000_000_000:,.{decimals}f}bn"
    if abs_value >= 1_000_000:
        return f"US${float(value) / 1_000_000:,.{decimals}f}m"
    return f"US${float(value):,.0f}"


def signed_usd(value: float, *, decimals: int = 1) -> str:
    """Format signed USD deltas without hiding direction."""
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{usd(value, decimals=decimals)}"


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


def display_label(name: str) -> str:
    """Map backend field names to professional UI labels."""
    return DISPLAY_LABELS.get(name, str(name).replace("_", " ").title())


def ordered_splits(values: list[str]) -> list[str]:
    """Return split values in TRAIN, VALIDATION, OOT order."""
    known = [split for split in SPLIT_ORDER if split in values]
    return known + [value for value in values if value not in known]


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
