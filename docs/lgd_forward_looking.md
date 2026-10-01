# Forward-Looking LGD Scenario Framework

This layer adds an interpretable macro overlay on top of the validated `lgd_v1_2`
structural LGD outputs. It does not rebuild LGD cashflows, modify PD/EAD/SICR,
or calculate ECL.

## Configuration

The main configuration file is `config/lgd_forward_looking.yaml`. It controls:

- parent LGD run;
- parent forward-looking macro scenario run;
- validation-only overlay calibration split;
- overlay clipping bounds;
- scenario horizon;
- macro features and economic stress weights;
- deterministic HPI and unemployment sensitivity shocks;
- artifact and model output locations.

The config shape is intentionally UI-ready for future Streamlit controls.

## Historical Regime Dataset

The historical regime dataset starts from resolved `lgd_v1_2` default episodes and
aggregates by default quarter and split. It includes:

- realized model-target LGD;
- structural predicted LGD from `lgd_v1_2`;
- cure rate;
- realized LGD for cured and non-cured episodes;
- rating/default-reason mix;
- LTV and current-UPB-to-original-UPB proxies;
- historical macro variables from the forward-looking PD run;
- historical systematic factor `Z`.

Historical fitting uses only information available for forward-looking modelling.
Scenario paths are not used during historical fitting.

## Overlay Specification

The selected overlay is a multiplicative macro stress adjustment:

```text
LGD_scenario = clip(LGD_v1_2_structural * macro_multiplier, 0, 1)
```

The macro multiplier is:

```text
macro_multiplier = exp(validation_intercept + slope * stress_index)
```

The stress index is a signed standardized score:

```text
+ unemployment_rate
- house_price_index_yoy
- real_gdp_yoy
+ mortgage_rate
- systematic_factor
```

The slope is estimated on TRAIN only and constrained to be non-negative. The
intercept is calibrated on VALIDATION only. OOT is held out for evaluation.

## Branch Decomposition

The overlay applies to the auditable `lgd_v1_2` branches:

```text
P(cure)
LGD | cure
LGD | non-cure
```

The adjusted expected LGD reconciles exactly:

```text
LGD = P(cure) * adjusted_cure_LGD
    + (1 - P(cure)) * adjusted_non_cure_LGD
```

The current v1 overlay keeps structural cure probability unchanged and applies
the macro multiplier to cure and non-cure severity. This avoids introducing a
second cure classifier while still allowing cure LGD and non-cure LGD to vary
by macro regime.

## Scenario Outputs

The layer applies the existing Base/Upside/Downside macro paths from the
forward-looking PD run. Outputs are produced by rating and scenario date:

- `lgd_base_structural`;
- `macro_overlay`;
- `lgd_scenario`;
- `scenario_weight`;
- `weighted_lgd`;
- probability-weighted LGD.

Scenario ordering is diagnosed explicitly. The framework does not force OOT O/E
to one and does not optimize on OOT.

## Limitations

Important limitations:

- The overlay uses quarterly portfolio/regime variation, so sample size is small.
- Structural cure probabilities are not re-estimated by macro scenario in v1.
- Macro coefficients are deliberately parsimonious and sign-constrained for
  interpretability.
- Residual LGD instability from `lgd_v1_2`, especially cure-LGD instability, is
  not eliminated by rebuilding borrower-level LGD.
- No ECL engine is implemented here.
