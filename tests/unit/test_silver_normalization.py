from __future__ import annotations

from datetime import date

import polars as pl

from ifrs9.transformations.missing_values import nullify_sentinels, status_expression
from ifrs9.transformations.normalization import (
    delinquency_months_expr,
    delinquency_status_type_expr,
    yyyymm_to_month_date,
)


def test_field_specific_sentinel_normalization() -> None:
    frame = pl.DataFrame({"_source_year": [2012, 2026], "vantagescore_4": [9999, 9999]})
    result = frame.with_columns(
        status_expression("vantagescore_4").alias("status"),
        nullify_sentinels("vantagescore_4"),
    )

    assert result["vantagescore_4"].to_list() == [None, None]
    assert result["status"].to_list() == ["STRUCTURAL_NOT_AVAILABLE", "NOT_AVAILABLE"]


def test_yyyymm_to_month_date() -> None:
    frame = pl.DataFrame({"period": ["201201", "", "202603"]}).with_columns(
        yyyymm_to_month_date("period")
    )

    assert frame["period"].to_list() == [date(2012, 1, 1), None, date(2026, 3, 1)]


def test_delinquency_status_mapping() -> None:
    frame = pl.DataFrame({"current_loan_delinquency_status": ["00", "03", "RA", "XX", "ZZ"]})
    result = frame.with_columns(delinquency_months_expr(), delinquency_status_type_expr())

    assert result["delinquency_months"].to_list() == [0, 3, None, None, None]
    assert result["delinquency_status_type"].to_list() == [
        "NUMERIC_DELINQUENCY_STATUS",
        "NUMERIC_DELINQUENCY_STATUS",
        "REO_ACQUISITION",
        "INITIAL_OR_UNKNOWN_STATUS",
        "UNKNOWN_SPECIAL_STATUS",
    ]
