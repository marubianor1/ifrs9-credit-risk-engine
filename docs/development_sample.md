# Temporal Development Sample

The development sample layer builds lightweight metadata for later traditional scorecard and PD model experiments. It does not train models, create model artifacts, or duplicate the 88-column Gold mart.

## Full-Horizon Rule

The 12-month binary PD development population requires a complete 12-month calendar outcome window. With reporting cutoff `2026-03-01`, the configured full-horizon cutoff is:

```text
as_of_date <= 2025-03-01
```

This rule is stricter than `target_12m_observable`. It intentionally excludes later observations even when a default or terminal event is observable, because those observations can be informatively observed and show unstable raw bad rates.

## Temporal Splits

Split windows are configured in `config/development_samples.yaml` and assigned strictly by `as_of_date`:

- TRAIN: `2012-01-01` to `2018-12-01`
- VALIDATION: `2019-01-01` to `2021-12-01`
- OOT: `2022-01-01` to `2025-03-01`

No random split is used. Random splitting would leak calendar regime information across train and validation samples and would understate temporal drift risk.

## Snapshot Frequency

Supported frequencies are:

- `monthly`
- `quarter_end`
- `year_end`

The default is `quarter_end`, meaning March, June, September, and December monthly observations. Rows are selected as actual point-in-time observations; there is no aggregation.

## Populations

The default population is `behavioural`: eligible loan-month observations with point-in-time features.

The `application` population interface is available and currently selects one early eligible observation per loan. This is a provisional interface only. A production application-scorecard methodology should later decide the exact observation anchor, such as acquisition month, first eligible performance month, or another documented application-time snapshot.

## Sampling

The default sampling strategy is `none`; all development observations are retained.

Optional non-default sampling strategies are available for later experiments:

- `random_nondefault`
- `stratified_nondefault`

Both retain all defaults, use a deterministic seed, calculate `sampling_probability`, and set `sample_weight = 1 / sampling_probability`. Stratification is by split and observation year at minimum.

## Repeated Loans

The default design is temporal portfolio validation, so the same loan can appear in multiple time windows. This is expected for behavioural PD modelling and is reported explicitly.

An optional `loan_disjoint` sensitivity mode is supported. It is not the default because forcing loans into only one split changes the temporal portfolio design.

## Outputs

Development metadata is written to:

```text
data/gold/freddie/development/
```

The output includes:

- `loan_id`
- `as_of_date`
- `split`
- `snapshot_frequency`
- `development_eligible_12m`
- `default_next_12m`
- `sample_selected`
- `sampling_probability`
- `sample_weight`

Artifacts are written to:

```text
artifacts/development/sample_profile.csv
artifacts/development/split_profile.csv
artifacts/development/development_manifest.json
```

Feature lists are recorded from the existing leakage-aware registry for the configured model use. Outcome-restricted and technical-only fields are not admitted as predictors.

## Production Profile

The default quarter-end behavioural sample contains:

- TRAIN: 4,021,497 rows, 342,168 loans, 15,242 defaults, bad rate 0.3790%.
- VALIDATION: 2,908,195 rows, 400,737 loans, 52,263 defaults, bad rate 1.7971%.
- OOT: 3,384,069 rows, 362,695 loans, 20,527 defaults, bad rate 0.6066%.

Monthly full-horizon development eligibility contains 30,655,079 rows and 656,871 loans. Quarter-end contains 10,313,761 rows and 655,536 loans. Year-end contains 2,612,639 rows and 625,145 loans.

Raw observable-target bad rates for 2025 and 2026 are not used for model development:

- 2025 raw observable bad rate: 1.6121%.
- 2026 raw observable bad rate: 4.8857%.
- 2025 full-horizon quarter-end development bad rate through March only: 0.6786%.
- 2026 contributes no full-horizon binary-development rows.
