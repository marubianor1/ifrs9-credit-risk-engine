# Logistic Scorecard Model

This layer implements traditional logistic scorecard experiments for the existing point-in-time development samples. It does not implement IFRS 9 PD calibration, TTC/PIT conversion, lifetime PD, LGD, EAD, SICR, staging, or ECL.

## Scope

The framework supports two populations:

- `behavioural`: point-in-time loan-month observations using behavioural features.
- `application`: one early eligible observation per loan using origination-safe features.

The application population is provisional. It is not yet equivalent to a true application-at-origination dataset, because the exact observation anchor still requires a methodological decision.

## Feature Selection

Feature lists are taken from the leakage-aware registry:

- `application_scoring` for application scorecards.
- `behavioural_scoring` for behavioural scorecards.

Outcome-restricted fields, technical fields, targets, split labels, sample weights, and future information are not admitted as predictors.

## WOE and IV

The scorecard pipeline fits supervised bins on TRAIN only.

Numerical features use quantile-based candidate bins, explicit missing bins, low-population bin merging, smoothing, and optional monotonic bad-rate enforcement. Categorical features use explicit missing categories and rare-category grouping.

For each bin the framework calculates:

- event and non-event counts;
- bad rate;
- weight of evidence;
- information value.

Bins are frozen after TRAIN fitting and applied unchanged to VALIDATION and OOT.

## Logistic Regression

The model is weighted logistic regression on WOE-transformed features. `sample_weight` is respected when non-default sampling is enabled. Coefficients, odds ratios, convergence-sensitive metrics, and sign checks versus WOE expectation are persisted.

Automatic validation/OOT-based feature selection is intentionally not implemented. Any future feature selection must be train-only and documented.

## Score Scaling

The default traditional scaling is:

- Base score: 600
- PDO: 50
- Base odds good:bad: 20:1

Higher scores indicate lower risk. Score bands are derived from TRAIN score quantiles and then applied unchanged to validation and OOT.

## Metrics and Stability

The framework reports, by split:

- ROC AUC and Gini;
- KS;
- PR-AUC;
- Brier score;
- log loss;
- calibration intercept and slope;
- observed and predicted bad rates;
- decile tables;
- score-band bad rates;
- ROC, calibration, and gains/lift tables.

Stability reporting includes score PSI, feature PSI, and yearly bad-rate, predicted-rate, Gini, and KS profiles. The 2019-2020 deterioration is visible in the yearly profiles and validation bad rates.

## Run Artifacts

Runs are persisted under:

```text
artifacts/models/scorecard/<run_id>/
models/scorecard/<run_id>/
```

The filesystem artifacts are complete and reproducible even when MLflow UI is not running. The package also provides reusable run-loading and run-comparison helpers for future Streamlit integration.

## Current Production Runs

Two baseline runs were trained:

- `behavioural_qe_v1`
- `application_qe_v1`

The behavioural model uses current balance, delinquency, delinquency history, months since delinquency, and current-to-original UPB. The application model uses origination credit score, DTI, LTV, CLTV, original UPB, and original interest rate.

These runs are benchmarks for future traditional scorecard development and not IFRS 9 calibrated PD models.
