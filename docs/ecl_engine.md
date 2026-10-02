# IFRS 9 Expected Credit Loss Engine

The ECL engine combines accepted PD, LGD, EAD, and staging artifacts without
refitting any model. The first production run is `ecl_v1`.

## Parent Runs

- PD: `forward_looking_pd_behavioural_qe_v1`;
- LGD: `lgd_v1_2`;
- EAD: `ead_v1`, contractual amortization method;
- Staging: `sicr_v1_1`.

## Formula

The reporting date is horizon month 0. Monthly ECL horizons are prospective:

```text
calendar_month = reporting_date + horizon_month
```

For example, with reporting date `2025-03-01`, horizon month 1 is `2025-04-01`.
The forward-looking PD artifact is built from quarterly scenario nodes. The
engine selects the first prospective scenario anchor on or after the reporting
date, such as `2025-06-30`, and uses the monthly marginal PD term structure
attached to that anchor. That anchor labels the macro scenario path; it does not
shift the ECL clock to June.

Stage 1 uses a 12-month horizon:

```text
ECL = sum(month 1..12) marginal_PD * LGD * EAD_month * discount_factor
```

Stage 2 uses the remaining contractual life, capped by the configured lifetime
horizon:

```text
ECL = sum(month 1..remaining_life) marginal_PD * LGD * EAD_month * discount_factor
```

Stage 3 does not use ordinary PD. PD is treated as 100%, and ECL is approximated
as current exposure times LGD. Detailed contractual default cash shortfall timing
is unavailable in the Freddie public performance data, so the current
implementation uses a month-zero expected-loss approximation.

## Scenarios

The engine calculates ECL separately for Upside, Base, and Downside using
scenario-specific forward-looking marginal PD term structures. LGD is structural
and macro-neutral because LGD macro challengers were rejected. Probability
weighted ECL is the scenario-weighted sum, and scenario weights must sum to one.

Quarterly macro scenario nodes are mapped to monthly ECL horizons by using the
documented monthly marginal PD curve produced for the selected prospective
scenario anchor. No realized macro observations after the reporting date are
introduced into the reporting-date ECL calculation.

## LGD Sensitivity

`lgd_method = structural_base` is the baseline. `downturn_sensitivity` applies
the validated `lgd_v1_2` downturn factors and is reported as a sensitivity, not a
fitted scenario LGD model.

## Outputs

Compact loan-level output is written to:

```text
data/gold/freddie/ecl/part-ecl.parquet
```

Diagnostics are written under:

```text
artifacts/ecl/<run_id>/
```

The engine also persists stage/rating/vintage/geography summaries, scenario
reconciliation, downturn sensitivity, explainability decomposition, and
alignment warnings.
