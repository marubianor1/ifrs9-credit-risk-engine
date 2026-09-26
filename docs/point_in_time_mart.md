# Point-in-Time Loan-Month Mart

The Gold Freddie Mac mart is the analytical foundation for later IFRS 9 work. It is built from the validated Silver origination and performance partitions and preserves a strict point-in-time observation unit:

```text
loan_id x as_of_date
```

`as_of_date` is the Silver performance `period`. The build does not fabricate missing months and does not create monthly observations for origination-only loans.

## Outputs

Gold partitions are written by vintage:

```text
data/gold/freddie/loan_static/vintage_year=YYYY/part-loan_static.parquet
data/gold/freddie/loan_month/vintage_year=YYYY/part-loan_month.parquet
```

The mart command is:

```bash
poetry run ifrs9 build-mart --year 2012 --force
poetry run ifrs9 build-mart --all --force
```

Build artifacts are written to `artifacts/mart/`:

- `mart_manifest.json`
- `loan_month_profile.csv`
- `feature_population.csv`
- `feature_distribution.csv`

These artifacts are execution outputs and are ignored by Git.

## Point-in-Time Rules

- Origination fields are treated as static application-time attributes.
- Current performance fields are only taken from the current `as_of_date`.
- Lag fields use exact calendar-month joins. If a loan skips a month, the lag for that missing calendar month is null.
- Cumulative history fields use rows at or before the current `as_of_date`.
- Rolling delinquency windows use calendar ranges ending at the current `as_of_date`.
- Outcome and recovery fields are retained for reporting and later LGD outcome construction, but the feature registry marks them as `OUTCOME_RESTRICTED`.
- Technical lineage columns are retained but marked as `TECHNICAL_ONLY`.

## DuckDB Views

`ifrs9.mart.views.register_gold_views` registers:

- `gold_loan_static`
- `gold_loan_month`
- `gold_loan_month_analytical`

The analytical view joins static origination attributes onto monthly observations by `loan_id` and `vintage_year`.

## Full Build Validation

The full build produced:

- Loan static rows: 712,500
- Unique static loans: 712,500
- Loan-month rows: 35,667,059
- Distinct monthly loans: 712,490
- Unique loan-month keys: 35,667,059
- Primary key duplicate count: 0
- Cross-vintage duplicate loan IDs: 0
- Observation date range: 2012-01-01 to 2026-03-01
- Gold disk footprint: 2.4G

The ten loans that exist only in Silver origination remain in `loan_static` and do not appear in `loan_month`.

## Current Scope

This layer does not define default, build targets, create model-ready samples, train models, assign IFRS 9 stages, or calculate ECL. Those steps remain downstream of the point-in-time analytical foundation.
