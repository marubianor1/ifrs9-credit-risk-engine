"""Gold loan static table construction."""

from __future__ import annotations

from pathlib import Path


def loan_static_sql(silver_path: Path, processed_at: str) -> str:
    """Return SQL for one Gold loan_static partition."""
    return f"""
    SELECT
        loan_id,
        _source_year AS vintage_year,
        origination_month AS origination_date,
        first_payment_date,
        maturity_date,
        credit_score AS original_credit_score,
        vantagescore_4,
        first_time_homebuyer_flag,
        original_debt_to_income_ratio AS original_dti,
        original_loan_to_value AS original_ltv,
        original_combined_loan_to_value AS original_cltv,
        original_upb,
        original_interest_rate,
        original_loan_term,
        number_of_borrowers,
        number_of_units,
        occupancy_status,
        property_type,
        loan_purpose,
        channel,
        mortgage_insurance_percentage,
        property_state,
        postal_code,
        property_valuation_method,
        property_valuation_method_status,
        interest_only_indicator,
        seller_name,
        special_eligibility_program,
        credit_score_status,
        vantagescore_4_status,
        original_debt_to_income_ratio_status,
        original_loan_to_value_status,
        original_combined_loan_to_value_status,
        _source_zip,
        _source_member,
        _ingested_at,
        _schema_version,
        _silver_processed_at,
        '{processed_at}' AS _gold_processed_at
    FROM read_parquet('{silver_path.as_posix()}')
    """

