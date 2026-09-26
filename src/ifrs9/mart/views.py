"""DuckDB analytical views for the Gold point-in-time mart."""

from __future__ import annotations

from pathlib import Path

import duckdb


def register_gold_views(con: duckdb.DuckDBPyConnection, repo_root: Path) -> None:
    """Register reusable DuckDB views over Gold Parquet partitions."""
    root = repo_root.as_posix()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW gold_loan_static AS
        SELECT * FROM read_parquet('{root}/data/gold/freddie/loan_static/*/*.parquet')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW gold_loan_month AS
        SELECT * FROM read_parquet('{root}/data/gold/freddie/loan_month/*/*.parquet')
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW gold_loan_month_analytical AS
        SELECT m.*, s.*
          EXCLUDE (loan_id, vintage_year, _gold_processed_at)
        FROM gold_loan_month m
        LEFT JOIN gold_loan_static s USING (loan_id, vintage_year)
        """
    )

