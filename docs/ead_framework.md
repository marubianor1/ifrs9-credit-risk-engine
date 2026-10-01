# IFRS 9 EAD Framework

This layer builds exposure at default data, baseline methods, and monthly EAD profiles for Freddie Mac amortizing mortgages. It does not implement SICR, staging, ECL, LGD, or revolving credit conversion factor logic.

Freddie Mac disclosure loans are amortizing mortgage exposures. The framework therefore focuses on current UPB, contractual amortization, and empirical future EAD at default. It does not model undrawn commitments or revolving utilization.

## Configuration

The main configuration file is `config/ead.yaml`. It controls:

- selected and candidate EAD methods;
- temporal development windows;
- maximum profile horizon;
- profile population cap;
- non-default exit diagnostic sample;
- amortization assumptions;
- empirical model bounds;
- safe point-in-time model features;
- artifact and model roots.

Artifacts are written under:

```text
artifacts/ead/<run_id>/
models/ead/<run_id>/
```

The CLI entry point is:

```bash
poetry run ifrs9 build-ead
```

## Population

The observation population uses eligible Gold loan-month records before default or terminal exit. Each row represents a point-in-time observation and retains safe information available at `as_of_date`, including:

- loan identifier and vintage;
- current actual UPB;
- current interest rate;
- remaining months to maturity;
- original term and original UPB;
- modification and assistance status;
- point-in-time delinquency and balance history;
- origination/static attributes;
- latest available behavioural PD rating at or before observation date.

Outcome fields, future event dates, realized EAD, EAD ratio, and method predictions are excluded from predictors.

## Target

For observations that subsequently default before non-default exit, the framework derives:

```text
ead_at_default
months_to_default
ead_ratio = ead_at_default / current_actual_upb
```

The default date comes from the existing validated default episode table. Default-entry EAD uses the centralized current-UPB then prior-month-UPB fallback helper, matching the already-established default-entry exposure logic for zero-balance credit-event rows.

LGD cashflow, recovery, expense, and actual-loss fields are not used.

Non-default terminal exits, including prepaid or matured loans, remain distinct from defaults. They are retained only for population/censoring diagnostics and are not treated as default observations.

## Methods

The framework compares three methods:

- `current_balance`: future EAD equals current actual UPB.
- `contractual_amortization`: future balance is projected from current UPB, current interest rate, remaining maturity, and standard amortization assumptions.
- `empirical_model`: an interpretable linear-regression model predicts bounded future EAD ratio using safe point-in-time features and the default horizon.

The empirical model fits TRAIN only. VALIDATION and OOT are scored without fitting to OOT.

## Profiles

Monthly lifetime EAD profiles are persisted with:

```text
month
ead_current_balance
ead_contractual
ead_modelled
```

The maximum horizon is configurable and defaults to 360 months. The default profile population is the latest eligible observation per loan, capped by configuration to keep artifacts tractable for downstream development. Target construction and backtesting are not profile-capped.

Profile validation checks that month 0 equals current exposure, contractual month 0 equals current exposure, modelled month 0 equals current exposure, and no EAD profile value is negative.

## Backtesting

Backtesting is performed on observable default rows. Each method is reported by TRAIN, VALIDATION, and OOT with:

- actual mean EAD;
- predicted mean EAD;
- observed-to-expected ratio;
- MAE;
- RMSE;
- MAPE where the denominator is safely above the configured floor.

Additional segment reports are persisted by:

- months-to-default band;
- rating;
- observation year;
- current LTV band;
- balance band.

## Limitations

Important limitations:

- Contractual amortization does not model prepayment, curtailments, forbearance capitalization, or modification-specific payment schedules.
- The empirical model is an interpretable baseline, not a final production EAD model.
- Non-default terminal exits are respected as censoring/exit events, but no competing-risk survival model is implemented in this layer.
- Freddie Mac disclosure data represents amortizing mortgage loans, not revolving facilities with undrawn commitments.
- No SICR, staging, LGD interaction, or ECL calculation is performed in this layer.
