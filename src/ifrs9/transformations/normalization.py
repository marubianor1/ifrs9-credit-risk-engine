"""Reusable Silver normalization helpers."""

from __future__ import annotations

import polars as pl


def yyyymm_to_month_date(field: str) -> pl.Expr:
    """Convert a Freddie Mac YYYYMM field to a first-of-month Date representation."""
    raw = pl.col(field).cast(pl.Utf8)
    cleaned = pl.when(raw.str.len_chars() == 6).then(raw + "01").otherwise(None)
    return cleaned.str.strptime(pl.Date, "%Y%m%d", strict=False).alias(field)


def delinquency_months_expr() -> pl.Expr:
    """Map numeric Freddie delinquency statuses to months past due, preserving special codes."""
    status = pl.col("current_loan_delinquency_status")
    return (
        pl.when(status.str.contains(r"^\d+$"))
        .then(status.cast(pl.Int64, strict=False))
        .otherwise(None)
        .alias("delinquency_months")
    )


def delinquency_status_type_expr() -> pl.Expr:
    """Classify Freddie delinquency status values without defining default."""
    status = pl.col("current_loan_delinquency_status")
    return (
        pl.when(status.str.contains(r"^\d+$"))
        .then(pl.lit("NUMERIC_DELINQUENCY_STATUS"))
        .when(status == "RA")
        .then(pl.lit("REO_ACQUISITION"))
        .when(status == "XX")
        .then(pl.lit("INITIAL_OR_UNKNOWN_STATUS"))
        .when(status.is_null() | (status == ""))
        .then(pl.lit("MISSING_STATUS"))
        .otherwise(pl.lit("UNKNOWN_SPECIAL_STATUS"))
        .alias("delinquency_status_type")
    )

