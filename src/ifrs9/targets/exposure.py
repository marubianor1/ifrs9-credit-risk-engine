"""Shared default-entry exposure expressions."""

from __future__ import annotations


def default_entry_ead_sql(alias: str = "m") -> str:
    """Return the established current-UPB then prior-month-UPB fallback expression."""
    return f"COALESCE(NULLIF({alias}.current_actual_upb, 0), {alias}.current_upb_lag_1)"


def default_entry_ead_source_sql(alias: str = "m") -> str:
    """Return the source label expression for default-entry EAD fallback."""
    return f"""
    CASE
        WHEN {alias}.current_actual_upb > 0 THEN 'current_actual_upb'
        WHEN {alias}.current_upb_lag_1 > 0 THEN 'prior_month_current_upb'
        ELSE 'unavailable'
    END
    """
