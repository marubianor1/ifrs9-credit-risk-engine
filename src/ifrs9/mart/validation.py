"""Validation helpers for the Gold point-in-time mart."""

from __future__ import annotations

from pathlib import Path

import duckdb

from ifrs9.mart.views import register_gold_views


def validate_gold_with_duckdb(repo_root: Path) -> dict[str, object]:
    """Validate persisted Gold Parquet partitions using DuckDB."""
    with duckdb.connect(database=":memory:") as con:
        register_gold_views(con, repo_root)
        static = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT loan_id) FROM gold_loan_static"
        ).fetchone()
        month = con.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT loan_id),
                   COUNT(DISTINCT loan_id || '|' || CAST(as_of_date AS VARCHAR)),
                   MIN(as_of_date),
                   MAX(as_of_date)
            FROM gold_loan_month
            """
        ).fetchone()
        duplicate_pk = con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT loan_id, as_of_date, COUNT(*) AS n
              FROM gold_loan_month
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        cross_vintage = con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT loan_id
              FROM gold_loan_static
              GROUP BY 1
              HAVING COUNT(DISTINCT vintage_year) > 1
            )
            """
        ).fetchone()[0]
        analytical_count = con.execute(
            "SELECT COUNT(*) FROM gold_loan_month_analytical"
        ).fetchone()[0]
    return {
        "loan_static_rows": static[0],
        "loan_static_unique_loans": static[1],
        "loan_month_rows": month[0],
        "loan_month_distinct_loans": month[1],
        "loan_month_unique_keys": month[2],
        "min_as_of_date": str(month[3]),
        "max_as_of_date": str(month[4]),
        "primary_key_duplicate_count": duplicate_pk,
        "cross_vintage_duplicate_loan_ids": cross_vintage,
        "analytical_view_rows": analytical_count,
    }
