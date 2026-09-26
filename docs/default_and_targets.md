# Default Definition and PD Targets

This layer builds default events and point-in-time PD target labels from the Gold Freddie Mac loan-month mart. It does not train models, create development samples, or define temporal train/validation/OOT splits.

## Default Definition

The default definition is configured in `config/default_definition.yaml`.

Version `v1` uses the following hierarchy:

1. `REO_ACQUISITION`: `current_loan_delinquency_status = RA`.
2. `CREDIT_EVENT_ZERO_BALANCE`: documented default-like zero-balance codes:
   - `02`: Third Party Sale
   - `03`: Short Sale or Charge Off
   - `09`: REO Disposition
3. `DELINQUENCY_3_PLUS`: numeric delinquency status at or above 3 months delinquent, used as a 90+ DPD proxy.

The following documented zero-balance exits are not defaults:

- `01`: Prepaid or Matured
- `15`: Whole Loan Sale
- `16`: Reperforming Loan Securitization
- `96`: Defect prior to other termination event

Loss, recovery, expense, and modification-cost fields are not used to determine delinquency default. They remain outcome-restricted fields for later LGD work.

Default flags are point-in-time: `default_flag` becomes true only in the monthly performance row where the delinquency threshold or documented credit event is observable.

## Derived Default Fields

The target build produces:

- `default_due_to_delinquency`
- `default_due_to_credit_event`
- `default_flag`
- `default_reason`
- `first_default_date`
- `months_to_first_default`
- `ever_default`

The default event table is written to:

```text
data/gold/freddie/targets/default_events/
```

## Cure and Redefault

The initial configurable cure rule requires the loan to return below the default threshold and remain non-defaulted for `cure.probation_months` consecutive observed monthly rows. The current probation length is 3 months.

Reporting gaps break the probation streak. Missing months are not fabricated.

The event table includes:

- `default_episode_id`
- `default_entry_date`
- `cure_date`
- `redefault_flag`
- `redefault_date`

Freddie Mac loan-level disclosure data is monthly and not a servicing system of record. The cure rule is therefore an analytical approximation over observed disclosure rows, not a claim to exact regulatory cure timing.

## Observation Eligibility

The target table preserves all Gold loan-month observations and adds eligibility fields:

- `eligible_for_12m_pd`
- `eligible_for_lifetime_pd`

Rows are not eligible after default has become observable or once a terminal zero-balance exit has occurred. Censored rows remain present and are explicitly marked rather than filtered away.

The reporting cutoff for version `v1` is `2026-03-01`.

## 12-Month PD Target

The 12-month target table is written to:

```text
data/gold/freddie/targets/pd_12m_targets/
```

For an eligible observation, `default_next_12m = 1` when the next qualifying default occurs after `as_of_date` and within the next 12 calendar months.

`default_next_12m = 0` only when:

- the full 12-month outcome window is observable through the reporting cutoff; or
- a legitimate non-default terminal exit occurs within the 12-month horizon before any default.

If the outcome cannot be determined because the reporting cutoff truncates the future window, `default_next_12m` is null and `target_12m_observable = false`.

The target uses calendar months, not row offsets.

## Lifetime and Survival Foundation

The target table includes fields for later lifetime PD or hazard modelling:

- `event_default`
- `event_prepayment_or_exit`
- `event_censored`
- `months_until_event`
- `event_type`

Competing non-default exits remain distinguishable from default events.

## Production Validation

The full build produced:

- Target rows: 35,667,059
- Unique target keys: 35,667,059
- Eligible 12-month observations: 34,439,082
- Positive 12-month targets: 274,454
- Negative 12-month targets: 30,577,383
- Censored 12-month targets: 3,587,245
- Loans ever defaulted: 24,171
- Default event episodes: 31,685
- Cures: 24,567
- Redefaults: 7,514

The three loans with reporting gaps produced 61 target rows. Forty-four were eligible and observable, all labelled non-default; three default events occurred on those loans. No missing months were generated.

## Artifacts

Execution artifacts are written under `artifacts/targets/`:

- `default_summary.csv`
- `target_observability.csv`
- `targets_manifest.json`
