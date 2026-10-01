# IFRS 9 SICR And Staging Framework

This layer assigns IFRS 9 stages from existing calibrated PD/rating outputs and validated default targets. It does not calculate ECL, lifetime losses, or LGD overlays.

## Configuration

The main configuration file is `config/sicr.yaml`. It controls:

- relative and absolute PD increase thresholds;
- rating downgrade threshold;
- delinquency backstop;
- optional other credit deterioration trigger;
- Stage 3 default handling;
- optional low-credit-risk exemption;
- Stage 2 cure probation;
- deterministic threshold sensitivity grid;
- staging and artifact output locations.

No SICR threshold is hardcoded in business logic.

## Reference Risk

The staging population is based on existing calibrated behavioural PD predictions. For each loan, the framework uses the earliest valid calibrated PD/rating observation as the initial-risk reference:

```text
sicr_reference_date
sicr_reference_pd
sicr_reference_rating
```

The framework reports the months between origination and the reference observation. It does not backfill current behaviour into origination. If no valid reference PD/rating exists, the origination baseline is marked unavailable and PD-change tests do not fire.

Validated active default loan-month rows are added separately for Stage 3 using the latest available PD/rating at or before each active default month. This avoids future leakage while allowing Stage 3 to persist through the default episode even when the PD scoring population excludes current-default observations.

## SICR Triggers

The framework creates independent trigger flags:

```text
sicr_relative_pd
sicr_absolute_pd
sicr_rating_downgrade
sicr_dpd_backstop
sicr_other_credit_deterioration
sicr_flag
sicr_reason
```

Default Stage 2 SICR logic is any enabled Stage 2 trigger. The delinquency backstop uses the current documented delinquency state only. A 30-DPD backstop is treated as Stage 2, not default.

The low-credit-risk exemption is optional. If disabled, it has no effect. If enabled, only configured high-quality rating grades may bypass Stage 2 SICR triggers; Stage 3 default still takes precedence.

## Stage Allocation

Stage allocation is:

```text
Stage 3: validated active default state
Stage 2: not Stage 3 and SICR after configured cure treatment
Stage 1: otherwise
```

The staging output keeps `default_entry_event` and `active_default_state` separately. Stage 3 starts at the validated default entry date and persists while the default episode is active, ending only at the validated cure date or when the loan leaves the observed performance history unresolved or terminal. Redefault starts a new active Stage 3 episode. Stage 3 takes precedence over every Stage 2 trigger.

## Cure And Transitions

Stage 3 cure relies on the existing validated default/cure framework: the staging layer does not invent a separate default cure policy.

Stage 2 to Stage 1 cure is configurable:

- `immediate`;
- `consecutive_months_without_sicr`.

The default is three consecutive months without SICR. Calendar gaps break the consecutive-month probation sequence.

## Outputs

Compact keyed staging data is written to:

```text
data/gold/freddie/staging/part-staging.parquet
```

The output keeps only staging keys, current PD/rating/reference fields, trigger flags, and final stage fields. It does not duplicate the full Gold mart.

Diagnostics are written under:

```text
artifacts/sicr/<run_id>/
```

Core diagnostics include:

- `stage_distribution.csv`;
- `stage_distribution_by_year_rating.csv`;
- `trigger_distribution.csv`;
- `stage_migrations.csv`;
- `threshold_sensitivity.csv`;
- `reference_pd_diagnostics.csv`;
- `stage3_state_audit.csv`;
- `sicr_trigger_exclusivity.csv`;
- `pd_relative_change_distribution.csv`;
- `reference_lag_profile.csv`.

## Sensitivity

The framework simulates a deterministic threshold grid without refitting PD:

- relative PD increase: `1.5x`, `2.0x`, `3.0x`;
- rating downgrade: `2`, `3`, `4` notches;
- DPD backstop: `1`, `2` months.

The output reports Stage 2 count, Stage 2 EAD, and migration rate for each scenario. This is designed to support future Streamlit sliders without moving business logic into Streamlit.

## Limitations

Important limitations:

- Behavioural PD reference risk starts at the earliest available scored observation, not true legal origination if no score exists at origination.
- Stage allocation depends on the quality and cadence of the existing PD scoring population.
- Behavioural SICR reference risk is a proxy for initial-recognition risk when no behavioural PD exists at legal origination.
- Stage 3 active default rows use latest prior PD/rating where exact month scoring is unavailable.
- No ECL, LGD forward-looking remediation, or lifetime loss engine is implemented here.
