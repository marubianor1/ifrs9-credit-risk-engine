# Silver Layer

The Silver layer standardizes Freddie Mac Bronze data into a cleaner analytical representation while preserving business meaning, row counts, source lineage, and auditability.

## Objective

Silver performs:

- business-safe type conversion;
- monthly date normalization;
- field-specific sentinel classification;
- selected normalized analytical values;
- source-semantic delinquency normalization;
- data-quality reporting.

Silver does not define default, derive modelling targets, calculate PD/LGD/EAD/SICR, assign stages, calculate ECL, join macroeconomic data, or create modelling features.

## Bronze Versus Silver

Bronze is source-faithful and preserves Freddie Mac physical layout values. Silver converts values into safer analytical types and classifies missing/sentinel semantics without filtering records.

Bronze and Silver row counts must match for each partition.

## Type Normalization

Numeric measures such as balances, rates, ratios, terms, recoveries, expenses, modification costs, and actual loss are retained as numeric values where safely represented.

Identifiers, categorical fields, postal codes, servicer/seller names, and mixed code/value fields remain strings.

## Date Normalization

Freddie Mac date fields are disclosed at monthly granularity as `YYYYMM`. Silver represents them as date values using the first day of the month. This day is technical only; the source granularity remains monthly.

Normalized monthly date fields include:

- `first_payment_date`
- `maturity_date`
- `period`
- `defect_settlement_date`
- `zero_balance_effective_date`
- `due_date_of_last_paid_installment`

`origination_month` is derived from the loan ID vintage quarter for consistency checks only.

## Sentinel Policy

Silver does not apply global replacements such as replacing every `999`, `9999`, `99`, `9`, `7`, `U`, `RA`, or `XX` with null.

Instead, sentinel handling is field-specific. Important fields receive companion status columns and are summarized in `artifacts/silver/normalization_summary.csv`.

Status labels include:

- `VALID_VALUE`
- `UNKNOWN`
- `NOT_APPLICABLE`
- `NOT_AVAILABLE`
- `NOT_DISCLOSED`
- `STRUCTURAL_NOT_AVAILABLE`
- `INVALID`

Historical structural non-availability is kept separate from ordinary missingness.

## Categorical Normalization

Silver preserves source categorical values when semantics are certain. It does not one-hot encode, collapse categories, or calculate WOE.

## Delinquency Normalization

`current_loan_delinquency_status` is preserved. Silver adds:

- `delinquency_months`: numeric delinquency month count only for numeric source statuses.
- `delinquency_status_type`: source-semantic classification for numeric statuses, `RA`, `XX`, missing, or unknown special values.

No default flag, stage, or SICR flag is created.

## Data Quality

Silver produces:

- `artifacts/silver/silver_manifest.json`
- `artifacts/silver/silver_data_quality.csv`
- `artifacts/silver/normalization_summary.csv`
- `artifacts/silver/historical_field_availability.csv`
- `artifacts/silver/performance_temporal_gaps.csv`
- `artifacts/silver/origination_without_performance.csv`
- `artifacts/silver/silver_summary.md`

DQ checks report structural errors, warnings, and informational metrics. Suspicious business values are reported, not corrected.

## Lineage

Silver preserves Bronze lineage fields:

- `_source_year`
- `_source_zip`
- `_source_member`
- `_ingested_at`
- `_schema_version`

Silver also adds `_silver_processed_at`.

Technical lineage fields must not be used as modelling features.

## Partitioning

Silver writes:

```text
data/silver/freddie/
  origination/vintage_year=YYYY/part-origination.parquet
  performance/vintage_year=YYYY/part-performance.parquet
```

Writes are idempotent by default. Existing partitions are skipped unless `--force` is used.

## Limitations

Silver is not a final analytical mart. Origination and performance remain separate. Field suitability for modelling must be assessed later using population, stability, leakage, and business validation checks.
