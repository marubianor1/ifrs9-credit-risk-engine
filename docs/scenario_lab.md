# IFRS 9 Scenario Lab Backend

The Scenario Lab is a reusable backend for sensitivity analysis on top of the
validated ECL engine. It does not refit PD, LGD, EAD, or staging models. It loads
accepted parent artifacts, applies deterministic in-memory shocks to copied
scenario state, and writes compact comparison outputs for future Streamlit
controls.

## Parent Runs

- PD: `forward_looking_pd_behavioural_qe_v1`
- LGD: `lgd_v1_2`
- EAD: `ead_v1`
- Staging: `sicr_v1_1`
- ECL: `ecl_v1`

## Execution Order

Scenario execution follows this order:

```text
portfolio shock
-> recalculated current risk state
-> SICR/staging
-> scenario PD
-> LGD assumption
-> EAD assumption
-> ECL
```

All overlays operate on a copied reporting-date state. Canonical Gold data and
parent model artifacts are not mutated.

## Supported Controls

Macro and PD controls:

- unemployment shock;
- HPI shock;
- GDP shock;
- mortgage-rate shock;
- scenario weights for Upside, Base, and Downside;
- optional rho override.

These controls are Scenario Lab stresses applied to the accepted
forward-looking PD term structures. They are not refitted macro satellite
models.

SICR and staging controls:

- relative PD threshold;
- absolute PD threshold;
- rating downgrade threshold;
- DPD backstop;
- Stage 2 cure probation.

Portfolio deterioration controls:

- deterministic rating downgrade share;
- 1-month delinquency shock share;
- 2-month delinquency shock share;
- 3+ delinquency/default shock share;
- deterministic seed.

LGD controls:

- structural base LGD;
- downturn LGD sensitivity;
- manual LGD overlay in percentage points.

Manual LGD overlays are user stresses and must not be presented as model
outputs.

EAD controls:

- stress-test EAD multiplier.

## Driver Waterfall

The Scenario Lab calculates a sequential waterfall:

```text
PD / macro effect
Stage migration effect
LGD effect
EAD effect
interaction / residual
```

The waterfall is order-dependent. It is designed for explainability and UI
diagnostics, not for Shapley-style attribution. Effects reconcile to total ECL
delta within numerical tolerance.

## Outputs

Each run writes compact artifacts under:

```text
artifacts/scenarios/<run_id>/
```

Files:

- `config.json`
- `summary.json`
- `stage_comparison.csv`
- `rating_comparison.csv`
- `driver_waterfall.csv`

The Scenario Lab does not persist another full 300k-loan dataset by default.

## API

```python
from ifrs9.scenario_lab import compare_scenarios, run_scenario

result = run_scenario(preset="mild_deterioration", run_id="scenario_mild_v1")
comparison = compare_scenarios(["scenario_mild_v1", "scenario_severe_v1"])
```

## Limitations

- Macro controls are deterministic stress overlays on accepted PD curves, not
  newly estimated macro models.
- Snapshot Stage 2 cure controls preserve already staged loans when probation is
  enabled, but full historical cure-path reallocation remains a future
  enhancement for richer UI diagnostics.
- Portfolio deterioration overlays are deterministic and illustrative; they do
  not infer borrower-level transition probabilities.
- The waterfall is sequential and order-dependent.
