# Point-in-Time Feature Catalog

The feature registry lives at `config/features/point_in_time_features.yaml`. It is intentionally leakage-aware: selector methods return only `SAFE_ORIGINATION` and `SAFE_AS_OF_DATE` features unless restricted fields are explicitly requested.

## Leakage Classes

- `SAFE_ORIGINATION`: available at application or acquisition time.
- `SAFE_AS_OF_DATE`: available at or before the monthly observation date.
- `OUTCOME_RESTRICTED`: terminal outcome, recovery, expense, or loss fields that must not be used as PD or behavioural predictors.
- `TECHNICAL_ONLY`: lineage and processing fields.

## Selectors

Use `FeatureRegistry.get_features_for(use)` or the convenience loader `load_default_registry(repo_root)`:

```python
from pathlib import Path
from ifrs9.mart.feature_registry import load_default_registry

registry = load_default_registry(Path.cwd())
pd_features = registry.get_features_for("pd")
```

The default selector excludes `OUTCOME_RESTRICTED` and `TECHNICAL_ONLY`. For audit reporting or LGD outcome construction, restricted fields can be requested explicitly with `include_restricted=True`.

## Registered Feature Groups

Application scoring currently uses origination-safe fields:

- `original_credit_score`
- `original_dti`
- `original_ltv`
- `original_cltv`
- `original_upb`
- `original_interest_rate`

Behavioural scoring and PD selectors add monthly and historical fields:

- `current_actual_upb`
- `delinquency_months`
- `max_delinquency_months_to_date`
- `months_since_last_delinquency`
- `current_upb_to_original_upb`

SICR selectors currently use delinquency state and history:

- `delinquency_months`
- `max_delinquency_months_to_date`
- `months_since_last_delinquency`

Restricted outcome fields retained in the mart but excluded from PD and behavioural selectors include:

- `actual_loss`
- `net_sale_proceeds`
- `total_expenses`

## Mart Feature Families

The `gold_loan_month` table contains broader analytical fields beyond the selector registry:

- Temporal depth: `loan_age_source`, `months_on_book_derived`, `months_since_origination`, `observed_months_to_date`, `calendar_months_since_first_observation`.
- Maturity: `remaining_months_to_maturity_source`, `months_to_contractual_maturity`.
- Calendar-aware lags: delinquency, UPB, and interest-rate lags at 1, 3, 6, and 12 month horizons where applicable.
- Balance evolution: current-to-original UPB, principal reduction, UPB changes, and percentage changes.
- Rate evolution: interest-rate change from origination and monthly/history lag deltas.
- Delinquency history: cumulative maximum, cumulative delinquent months, event counts, months since last delinquency, rolling 3/6/12 month maxima, and rolling delinquent-month counts.
- Modification and assistance history: ever-modified, months since last modification, ever-assistance, and current assistance flags.
- Reporting gaps: `has_reporting_gap_to_date`.
- Restricted outcomes: recoveries, expenses, losses, and modification cost fields.
- Technical lineage: source archive member, schema version, Silver timestamp, and Gold timestamp.

The first production registry version contains 15 selector-managed fields. The monthly mart itself contains 88 columns, including identifiers, lineage, restricted outcomes, and operational fields that are not all exposed as default modelling predictors.
