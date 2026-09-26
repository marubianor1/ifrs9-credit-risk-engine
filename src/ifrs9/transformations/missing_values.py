"""Field-specific missing and sentinel handling for Silver transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

MissingStatus = Literal[
    "VALID_VALUE",
    "UNKNOWN",
    "NOT_APPLICABLE",
    "NOT_AVAILABLE",
    "NOT_DISCLOSED",
    "STRUCTURAL_NOT_AVAILABLE",
    "INVALID",
]


@dataclass(frozen=True)
class SentinelRule:
    """Normalization rule for a field-specific source sentinel."""

    field: str
    sentinels: dict[str, MissingStatus]
    structural_year_before: int | None = None


SENTINEL_RULES: dict[str, SentinelRule] = {
    "credit_score": SentinelRule("credit_score", {"9999": "NOT_AVAILABLE"}),
    "vantagescore_4": SentinelRule("vantagescore_4", {"9999": "NOT_AVAILABLE"}, 2026),
    "original_debt_to_income_ratio": SentinelRule(
        "original_debt_to_income_ratio", {"999": "NOT_AVAILABLE"}
    ),
    "original_loan_to_value": SentinelRule("original_loan_to_value", {"999": "NOT_AVAILABLE"}),
    "original_combined_loan_to_value": SentinelRule(
        "original_combined_loan_to_value", {"999": "NOT_AVAILABLE"}
    ),
    "property_valuation_method": SentinelRule(
        "property_valuation_method",
        {"": "STRUCTURAL_NOT_AVAILABLE", "7": "NOT_AVAILABLE"},
        structural_year_before=2020,
    ),
    "estimated_loan_to_value": SentinelRule(
        "estimated_loan_to_value",
        {"999": "UNKNOWN"},
        structural_year_before=2017,
    ),
    "net_sale_proceeds": SentinelRule("net_sale_proceeds", {"U": "UNKNOWN", "": "NOT_APPLICABLE"}),
    "mortgage_insurance_cancellation_indicator": SentinelRule(
        "mortgage_insurance_cancellation_indicator",
        {"7": "NOT_APPLICABLE", "9": "NOT_DISCLOSED"},
    ),
}


def status_expression(field: str, source_year_field: str = "_source_year") -> pl.Expr:
    """Return a Polars expression classifying field-specific missing semantics."""
    rule = SENTINEL_RULES[field]
    value = pl.col(field).cast(pl.Utf8)
    expr = pl.when(pl.col(field).is_null()).then(pl.lit("NOT_APPLICABLE"))
    for sentinel, status in rule.sentinels.items():
        status_value = (
            "STRUCTURAL_NOT_AVAILABLE"
            if rule.structural_year_before is not None
            else status
        )
        condition = value == sentinel
        if rule.structural_year_before is not None:
            condition = condition & (pl.col(source_year_field) < rule.structural_year_before)
        expr = expr.when(condition).then(pl.lit(status_value))
        if rule.structural_year_before is not None:
            expr = expr.when(value == sentinel).then(pl.lit(status))
    return expr.otherwise(pl.lit("VALID_VALUE"))


def nullify_sentinels(field: str) -> pl.Expr:
    """Return an expression that nulls only field-specific sentinel values."""
    rule = SENTINEL_RULES[field]
    value = pl.col(field).cast(pl.Utf8)
    condition = pl.col(field).is_null()
    for sentinel in rule.sentinels:
        condition = condition | (value == sentinel)
    return pl.when(condition).then(None).otherwise(pl.col(field)).alias(field)
