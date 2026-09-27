# Calibrated PD Framework

This layer converts existing scorecard outputs into reusable probability of default artifacts for IFRS 9 development. It keeps the architecture explicit:

```text
Score / ranking
-> 12M raw PD
-> calibrated PD
-> rating master scale
-> TTC anchor
-> lifetime term structure
```

The current production population is `behavioural_qe_v1`. The application scorecard remains available as a secondary benchmark, but it is not the primary PD population.

This layer does not implement macroeconomic adjustment, Vasicek overlays, LGD, EAD, SICR, staging, or ECL.

## Configuration

The main configuration file is `config/pd.yaml`. It controls:

- parent scorecard run IDs;
- calibration method;
- rating labels and minimum grade thresholds;
- TTC anchor window;
- lifetime horizon and smoothing;
- transition frequencies;
- artifact locations.

Artifacts are written to:

```text
artifacts/pd/<run_id>/
models/pd/<run_id>/
```

The framework exposes:

```python
run_pd_framework(config) -> PDRunResult
```

The command-line entry point is:

```bash
poetry run ifrs9 build-pd --scorecard-run behavioural_qe_v1
```

Use `--run-id` for deterministic artifact names and `--force` to overwrite an existing PD run.

## Calibration

The scorecard model is fitted upstream on TRAIN. The PD calibration layer fits only on the configured calibration split, currently VALIDATION. OOT rows are never used for fitting.

Supported methods are:

- `none`: pass through the raw scorecard PD.
- `logistic_recalibration`: fit a logistic model on the logit of raw PD.
- `isotonic`: fit monotonic isotonic calibration.

The default method is `logistic_recalibration`.

Outputs include:

- `pd_raw`;
- `pd_calibrated_12m`;
- raw and calibrated metrics for TRAIN, VALIDATION, and OOT;
- calibration curve;
- expected calibration error.

The logistic recalibration coefficient is constrained to preserve risk ordering if the fitted slope would be negative. Isotonic calibration is fitted as an increasing function.

## Rating Master Scale

The rating scale is ordinal. The default labels are `R1` through `R10`, where higher grades represent higher risk.

Grade boundaries are derived from TRAIN calibrated PDs only and then applied unchanged to all splits. The output includes:

- persisted boundaries;
- population;
- weighted defaults;
- observed bad rate;
- mean calibrated PD;
- minimum and maximum calibrated PD.

The configuration includes minimum population and minimum default-count thresholds for governance review. These thresholds are reported through the grade summaries and backtesting flags; they do not use OOT to alter cutoffs.

## TTC Anchor

The TTC anchor is a transparent long-run historical baseline from observable 12-month outcomes. The initial window is:

```text
2012-01-01 to 2024-03-01
```

This excludes 2025-2026 censored periods by configuration. The initial weighting method is observation-weighted, using the row-level sample weights already carried by the scorecard population.

The outputs are:

- `portfolio_ttc_pd`;
- `rating_ttc_pd`.

No macro adjustment is applied in this layer.

## PIT Diagnostics

The framework produces realized and predicted diagnostics by observation year:

- observed 12-month default rate;
- mean calibrated PD;
- TTC PD;
- PIT-to-TTC ratio.

These diagnostics are descriptive. The calibrated scorecard PD is not labelled as macro-adjusted PIT PD.

## Lifetime Term Structure

Lifetime PD curves are estimated from the default and censoring event foundation. The current approach is a discrete monthly hazard model by current rating.

For each rating and month, the framework estimates:

- conditional PD or hazard;
- marginal PD;
- cumulative PD;
- survival probability.

Right censoring is handled through the at-risk denominator. Non-default terminal exits remain available as separate event flags in the event foundation and are not treated as defaults.

The default maximum horizon is 360 months. Smoothing is configurable and defaults to 0.5.

## Reconciliation

When enabled, the lifetime curve is calibrated so the first 12 months reconcile to the rating-level 12-month calibrated PD. The implementation solves for a single hazard scaling factor that matches the target cumulative 12-month PD after monthly survival compounding.

This is not an annual PD multiplied by maturity. The curve remains a monthly conditional hazard term structure.

## Transitions

The framework produces 1-month and 12-month transition matrices from current ratings and future rating states where the data frequency supports the horizon.

The default state is absorbing. Transition summaries report:

- migration rate;
- upgrade rate;
- downgrade rate;
- default transition rate.

Transitions are diagnostic only and are not used for SICR in this layer.

## Backtesting

Backtesting is produced by rating and observation year, and at portfolio split level.

For each rating-year the output includes:

- observed defaults;
- expected defaults;
- observed-to-expected ratio;
- binomial confidence interval;
- Brier score;
- insufficient-default flag.

Portfolio-level split backtesting is reported for TRAIN, VALIDATION, and OOT.
