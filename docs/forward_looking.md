# Forward-Looking PD Framework

This layer converts calibrated TTC PD outputs into forward-looking scenario-conditioned PD curves. It is built on top of parent PD run `pd_behavioural_qe_v1`.

The architecture is:

```text
TTC PD
-> systematic macro factor
-> scenario-conditioned PIT PD
-> Base / Upside / Downside PD curves
-> probability-weighted PD
```

This layer does not implement LGD, EAD, SICR, staging, or ECL.

## Configuration

The main configuration file is `config/forward_looking.yaml`. It controls:

- parent PD run;
- forward-looking method;
- macro source and series;
- Vasicek asset-correlation bounds or override;
- satellite model settings;
- scenario weights and shocks;
- lifetime adjustment method;
- artifact paths.

Artifacts are written under:

```text
artifacts/forward_looking/<run_id>/
models/forward_looking/<run_id>/
```

The command-line entry point is:

```bash
poetry run ifrs9 build-forward-looking --pd-run pd_behavioural_qe_v1
```

Use `--run-id` for deterministic artifact names and `--force` to overwrite an existing run.

## Macro Data

The initial public macro pipeline uses FRED CSV endpoints and caches raw downloads under `data/raw/macro/`, which is ignored by Git.

Initial series:

- `UNRATE`: unemployment rate.
- `CSUSHPINSA`: S&P CoreLogic Case-Shiller U.S. National Home Price Index, transformed to year-over-year growth.
- `GDPC1`: real gross domestic product, transformed to year-over-year growth.
- `MORTGAGE30US`: 30-year fixed mortgage average rate.

All series are aligned to quarterly dates because the behavioural modelling population is quarter-end based. Monthly and weekly series are averaged to quarter end. Quarterly GDP is carried as the quarter-end observation. Derived features include unemployment-rate change and mortgage-rate change.

Downloaded macro data is not committed.

## Vasicek Method

The default method is one-factor Vasicek:

```text
PD_PIT(r,t) =
Phi[
  (Phi^-1(PD_TTC(r)) - sqrt(rho) * Z_t)
  / sqrt(1-rho)
]
```

The sign convention is:

```text
higher Z = better economy
```

A worse economy has a lower systematic factor and therefore a higher PIT PD.

Historical `Z_t` is inferred from observed portfolio default experience and portfolio TTC PD:

```text
Z_t =
(Phi^-1(PD_TTC) - sqrt(1-rho) * Phi^-1(observed_default_rate_t))
/ sqrt(rho)
```

Observed rates are clipped away from 0 and 1 for numerical stability.

`rho` can be supplied directly through configuration. If no override is supplied, the framework searches configured bounds and selects the value whose inferred historical factor volatility is closest to one.

## Macro Factor Model

The framework fits an interpretable linear model from macro features to the historical systematic factor. For the Vasicek factor model, features are oriented by economic strength and constrained so better macro conditions imply a higher `Z`. This preserves the convention that better macro conditions produce lower PIT PD.

Candidate predictors are configured and currently include:

- unemployment rate;
- unemployment-rate change;
- HPI year-over-year growth;
- real GDP year-over-year growth;
- mortgage rate;
- mortgage-rate change.

The model is fitted using historical macro and default observations only. Scenario paths are not used in estimation.

Persisted diagnostics include:

- coefficients;
- residuals;
- RMSE and MAE;
- lag-1 residual autocorrelation;
- variance inflation factors.

## Satellite Challenger

The challenger `macro_satellite` model predicts portfolio default risk directly from macro variables using a transparent regularized linear model on the logit of observed default rates.

It is persisted and backtested beside the Vasicek framework. It is not automatically selected as champion.

## Scenarios

The framework supports:

- `historical_shock`;
- `custom`.

The default configured scenarios are:

- `BASE`;
- `UPSIDE`;
- `DOWNSIDE`.

Scenario weights must sum to one. Weights can be changed and scenario PDs can be reweighted without refitting historical models.

Configured shocks are applied as scenario controls rather than hardcoded production assumptions. They are designed for future Streamlit controls.

## Scenario PD Curves

For each rating, scenario, and scenario quarter, the framework produces:

- TTC PD;
- systematic factor;
- scenario-conditioned 12-month PIT PD;
- scenario weight;
- weighted 12-month PD.

Probability-weighted PD is:

```text
weighted_PD = sum(scenario_weight * scenario_PD)
```

## Lifetime Adjustment

Forward-looking lifetime curves are adjusted through monthly conditional hazards. For each rating and scenario quarter, the framework solves for a single hazard scaling factor so the first 12 months reconcile to the scenario-conditioned 12-month PIT PD.

The method preserves:

- `0 <= marginal PD <= 1`;
- `0 <= cumulative PD <= 1`;
- cumulative PD is non-decreasing;
- survival probability is non-increasing.

This is not a multiplication of annual PD by maturity.

## Backtesting

Historical quarterly backtesting reports:

- observed default rate;
- TTC PD;
- Vasicek PIT PD;
- satellite PIT PD;
- residuals;
- RMSE, MAE, and Brier-style error metrics.

The 2019-2021 stress period is persisted separately for explicit review.

## Limitations

The current framework is a transparent portfolio project layer, not a production IFRS 9 overlay. Important limitations:

- limited quarterly history constrains macro model stability;
- macro data revisions are not versioned beyond cached raw files and FRED series identifiers;
- scenario shocks are simple configurable paths, not official economic forecasts;
- rating-level PIT PDs use TTC rating anchors and the common systematic factor;
- no LGD, EAD, SICR, staging, or ECL calculation is performed here.
