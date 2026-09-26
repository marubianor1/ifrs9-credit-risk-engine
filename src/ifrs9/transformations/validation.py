"""Data-quality checks for Silver transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl


@dataclass(frozen=True)
class SilverDQMetrics:
    """Per-year Silver data-quality metrics."""

    year: int
    dataset: str
    severity: str
    check_name: str
    value: int | float | str | None

    def as_dict(self) -> dict[str, int | float | str | None]:
        """Return a serializable representation."""
        return asdict(self)


def origination_dq(frame: pl.DataFrame, year: int) -> list[SilverDQMetrics]:
    """Calculate origination DQ checks."""
    return [
        SilverDQMetrics(year, "origination", "INFO", "row_count", frame.height),
        SilverDQMetrics(
            year, "origination", "INFO", "unique_loan_count", frame["loan_id"].n_unique()
        ),
        SilverDQMetrics(
            year,
            "origination",
            "ERROR",
            "duplicate_loan_ids",
            frame.height - frame["loan_id"].n_unique(),
        ),
        SilverDQMetrics(
            year,
            "origination",
            "ERROR",
            "null_loan_ids",
            frame.filter(pl.col("loan_id").is_null() | (pl.col("loan_id") == "")).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_original_upb",
            frame.filter(
                pl.col("original_upb").is_not_null() & (pl.col("original_upb") <= 0)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_credit_score",
            frame.filter(
                pl.col("credit_score").is_not_null()
                & ~pl.col("credit_score").is_between(300, 850)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_dti",
            frame.filter(
                pl.col("original_debt_to_income_ratio").is_not_null()
                & ~pl.col("original_debt_to_income_ratio").is_between(1, 65)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_ltv",
            frame.filter(
                pl.col("original_loan_to_value").is_not_null()
                & ~pl.col("original_loan_to_value").is_between(1, 998)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_cltv",
            frame.filter(
                pl.col("original_combined_loan_to_value").is_not_null()
                & ~pl.col("original_combined_loan_to_value").is_between(1, 998)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "invalid_first_payment_date",
            frame.filter(pl.col("first_payment_date").is_null()).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "first_payment_before_origination",
            frame.filter(pl.col("first_payment_date") < pl.col("origination_month")).height,
        ),
        SilverDQMetrics(
            year,
            "origination",
            "WARNING",
            "maturity_before_first_payment",
            frame.filter(pl.col("maturity_date") < pl.col("first_payment_date")).height,
        ),
    ]


def performance_dq(frame: pl.DataFrame, year: int) -> list[SilverDQMetrics]:
    """Calculate performance DQ checks."""
    unique_pairs = frame.select(pl.struct("loan_id", "period").n_unique()).item()
    return [
        SilverDQMetrics(year, "performance", "INFO", "row_count", frame.height),
        SilverDQMetrics(
            year, "performance", "INFO", "unique_loan_count", frame["loan_id"].n_unique()
        ),
        SilverDQMetrics(
            year,
            "performance",
            "ERROR",
            "duplicate_loan_periods",
            frame.height - unique_pairs,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "ERROR",
            "null_loan_ids",
            frame.filter(pl.col("loan_id").is_null() | (pl.col("loan_id") == "")).height,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "ERROR",
            "null_period",
            frame.filter(pl.col("period").is_null()).height,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "WARNING",
            "negative_current_upb",
            frame.filter(
                pl.col("current_actual_upb").is_not_null()
                & (pl.col("current_actual_upb") < 0)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "WARNING",
            "invalid_current_interest_rate",
            frame.filter(
                pl.col("current_interest_rate").is_not_null()
                & ~pl.col("current_interest_rate").is_between(0, 30)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "WARNING",
            "invalid_remaining_maturity",
            frame.filter(
                pl.col("remaining_months_to_legal_maturity").is_not_null()
                & (pl.col("remaining_months_to_legal_maturity") < 0)
            ).height,
        ),
        SilverDQMetrics(
            year,
            "performance",
            "WARNING",
            "unknown_delinquency_status",
            frame.filter(pl.col("delinquency_status_type") == "UNKNOWN_SPECIAL_STATUS").height,
        ),
    ]
