# IFRS 9 LGD Framework

This layer builds loss given default data and baseline models from existing default episodes and Freddie Mac recovery/loss fields. It does not implement EAD forecasting, SICR, staging, or ECL.

## Configuration

The main configuration file is `config/lgd.yaml`. It controls:

- default definition source;
- resolution horizon;
- cure probation reference;
- discounting method;
- temporal development cutoffs;
- model target bounds;
- segmentation variables;
- cure and severity model settings;
- downturn overlay method;
- feature lists and artifact roots.

Artifacts are written under:

```text
artifacts/lgd/<run_id>/
models/lgd/<run_id>/
```

The CLI entry point is:

```bash
poetry run ifrs9 build-lgd
```

## Population

The population is one row per existing default episode from `data/gold/freddie/targets/default_events/`. The framework does not redefine default.

At default entry it captures:

- loan and default episode identifiers;
- default date and reason;
- EAD at default;
- current interest rate at default;
- latest available behavioural PD rating at or before default;
- origination/static attributes;
- point-in-time behavioural variables at default.

Default-entry EAD uses current actual UPB at default. If default is first observed on a zero-balance credit-event row with zero UPB, the framework falls back to prior-month current UPB and persists `ead_at_default_source`.

## Resolution Outcomes

Resolution is assessed within the configured horizon. Current mappings are:

- `CURE`: existing cure date occurs before any terminal zero-balance event.
- `REO_FORECLOSURE`: zero-balance code `09`.
- `SHORT_SALE_OR_CHARGE_OFF`: zero-balance code `03`.
- `THIRD_PARTY_SALE`: zero-balance code `02`.
- `OTHER_DOCUMENTED_TERMINAL`: documented terminal code `01`, `15`, `16`, or `96`.
- `UNRESOLVED`: no cure or documented terminal outcome within the horizon.

Unresolved episodes remain in the target dataset but are excluded from supervised realized-LGD model fitting.

## Cashflow Components

The framework reconstructs auditable economic-loss components using canonical Gold mart fields:

- `net_sale_proceeds`;
- `mi_recoveries`;
- `non_mi_recoveries`;
- `total_expenses`;
- `cumulative_modification_costs`;
- `bankruptcy_cramdown_costs`;
- `actual_loss`.

`zero_balance_removal_upb` and `delinquent_accrued_interest` are documented in the source schema but are not currently carried into the Gold mart. The LGD builder joins the terminal Silver performance row to retrieve those terminal values without rebuilding the mart.

`actual_loss` is used for reconciliation only. It is not a model predictor.

## Economic LGD

Recoveries and costs are discounted from resolution date to default date using the current interest rate at default as an effective-rate proxy. If that rate is missing, the framework falls back to the original interest rate and then zero.

Freddie recovery fields are signed terminal values. Recovery credits are therefore added as signed amounts rather than subtracted as if they were positive proceeds. This prevents the double-negative error that inflated `lgd_v1` reconstructed loss.

The economic loss formula is:

```text
economic_loss = exposure basis + signed discounted recoveries + discounted costs
LGD = economic_loss / exposure basis
```

Raw LGD is not clipped for diagnostics. The model target is bounded using configurable target bounds, currently 0 to 1.

## Cure Treatment

The framework uses a two-stage structure:

```text
P(cure | default)
+
LGD severity conditional on non-cure
```

Cured episodes use observed cashflow/loss information when available. Where cured episodes have no recovery/loss evidence in Freddie fields, the framework applies the explicit configured fallback `cure_fallback_lgd`, currently 0.0, and flags `cure_lgd_fallback_used`.

Expected LGD is:

```text
ELGD =
P(cure) * LGD_cure
+ (1 - P(cure)) * LGD_non_cure
```

The framework persists `component_decomposition.csv` to reconcile this equation by split.
The decomposition reports observed and predicted cure probability, cure LGD,
non-cure probability, non-cure LGD, the combined expected LGD, and the calibration
factor used for the selected specification.

Run `lgd_v1_3` replaces the constant cure-LGD assumption with a collateral-driven
cure severity challenger set. The cure-severity population is limited to resolved
cured default episodes. Fitting uses TRAIN cured episodes only; VALIDATION is used
for calibration and model choice; OOT remains held out for final evaluation.

The cure-LGD challenger set includes:

- `segment_mean_baseline`;
- `bounded_regression`;
- `fractional_logit_style_regression`.

Candidate predictors are safe point-in-time/default-entry fields, including
estimated LTV at default, original LTV/CLTV, EAD at default, loan age,
interest-rate fields, delinquency history, rating, default reason, and
modification/assistance flags. Outcome, recovery, resolution, and cashflow fields
are excluded.

The current-LTV proxy is `estimated_loan_to_value_at_default`. Missingness and
cure LGD by LTV band are persisted in `cure_lgd_ltv_audit.csv`; challenger
metrics are persisted in `cure_lgd_model_comparison.csv`.

Run `lgd_fl_v2` applies scenario HPI through collateral leverage rather than a
direct macro-LGD regression:

```text
cumulative_HPI_change = product(1 + HPI_yoy / 100 / 4) - 1
stressed_LTV = current_LTV / (1 + cumulative_HPI_change)
```

The stressed LTV is passed through the selected cure-LGD model. Cure probability
and non-cure severity remain inherited from `lgd_v1_3`. This produces
Base/Upside/Downside and probability-weighted LGD without calculating ECL.

## Models

The cure model challenger set keeps the interpretable logistic baseline and adds:

- validation-recalibrated logistic regression;
- recency-weighted logistic regression with validation-only calibration.

The non-cure severity challenger set includes:

- the current linear-regression baseline;
- a bounded fractional-logit challenger fitted through a logit-transformed target;
- a rating-segment-calibrated challenger.

Temporal training weights support:

- `none`;
- `linear_recency`;
- `exponential_recency`.

Decay choices are selected using TRAIN and VALIDATION only. OOT is scored only after
selection as an unbiased holdout.

Models fit TRAIN only. VALIDATION and OOT are scored but not used for fitting. Outcome and recovery fields are excluded from predictors.

The initial feature set is deliberately conservative and uses point-in-time default-entry variables plus safe static origination attributes.

## Segmentation

Segmentation reporting is configurable. The initial dimensions are:

- rating at default;
- current LTV band;
- default reason.

Segments are reported with sample counts and an acceptance flag based on the configured minimum resolved-default count.

## Downturn LGD

The downturn overlay framework supports:

- `none`;
- `historical_stress`;
- `quantile_overlay`.

The current run uses `historical_stress`, comparing realized LGD by rating during the configured stress window against baseline realized LGD. The overlay is floored at 1.0 and persisted as `lgd_downturn` beside `lgd_base`.

No regulatory downturn scalar is invented.

## Backtesting

Backtesting reports by split:

- resolved defaults;
- unresolved defaults;
- cure rate;
- realized mean and median LGD;
- predicted LGD;
- observed-to-expected ratio;
- MAE and RMSE.

Additional outputs compare observed and predicted LGD by rating, default year, and resolution type. Recovery timing is reported by split and resolution type.

Model comparison outputs are persisted as:

- `cure_model_comparison.csv`;
- `severity_model_comparison.csv`;
- `combined_backtest.csv`;
- `predictor_drift.csv`.

Predictor drift is measured against TRAIN using PSI, missingness, summary
statistics, and category mix. Resolution mix is included as an outcome diagnostic
only, not as a predictor.

## Reconciliation

The framework persists a Freddie-style reconstruction separately from economic LGD:

```text
zero_balance_removal_upb
+ signed net sale proceeds
+ signed MI recoveries
+ signed non-MI recoveries
+ total expenses
+ delinquent accrued interest
= Freddie-style reconstructed actual loss
```

`actual_loss` remains validation-only and is not used as a predictor or model target input.

## Limitations

Important limitations:

- Freddie disclosure data is not a full servicing cashflow ledger.
- The economic LGD target may differ from Freddie `actual_loss` because it includes modelling choices such as discounting and additional economic cost components.
- Cure-loss observations are sparse; many cured episodes use the explicit configured fallback.
- The current models are interpretable baselines, not final production LGD models.
- No EAD forecasting, SICR, staging, or ECL is performed in this layer.
