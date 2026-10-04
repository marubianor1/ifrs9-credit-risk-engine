# Streamlit Portfolio App

The Streamlit app is a presentation layer for the existing IFRS 9 artifacts and backend APIs. It must not contain model business logic; pages call thin services under `app/services/`, and services either load persisted artifacts or invoke existing backend functions in `src/ifrs9`.

## Launch

Run the app from the repository root:

```bash
poetry run streamlit run app/streamlit_app.py
```

The app defaults to existing demo runs and does not automatically trigger expensive model rebuilds.

## Navigation

The final app exposes the following page order:

- Overview
- Scoring Models
- PD
- LGD
- EAD
- SICR & Staging
- ECL
- Scenario Lab
- Model Monitoring
- Report Generator
- About / Methodology

## Risk Pages

### LGD

The LGD page reads `lgd_v1_2` as the production/project baseline and allows review of `lgd_v1_3`, `lgd_fl_v1`, and `lgd_fl_v2` as rejected challengers. It shows default and resolution population, cure rate, realized versus predicted LGD, cure and non-cure decomposition, split-level O/E, LGD by rating and default year, downturn factors, recovery timing, and challenger diagnostics. LGD is not refit from the UI.

### EAD

The EAD page reads `ead_v1`, highlights `contractual_amortization` as the selected baseline method, and shows EAD ratio distribution, method backtests, segment results, target reconciliation, profile validation, and sample lifetime EAD profiles. The `Run EAD` control calls the existing backend only after an explicit button click.

### SICR & Staging

The staging page reads `sicr_v1_1`, shows Stage 1/2/3 counts and EAD, trigger distribution, trigger overlap, stage migrations, Stage 3 duration and cure diagnostics, reference PD methodology, and behavioural reference lag. Threshold controls are converted by the staging service and evaluated only after `Run Staging Simulation`; the page compares baseline and simulated stage rows/EAD and does not calculate ECL.

### ECL

The ECL page reads `ecl_v1` and shows Total EAD, weighted ECL, coverage ratio, Stage 1/2/3 ECL, EAD/ECL by stage, ECL by rating, Base/Upside/Downside totals, coverage by stage, Stage 1 12-month versus Stage 2 lifetime output, downturn-LGD sensitivity, and alignment warnings. The page displays the key methodological limitations: structural macro-neutral LGD, Stage 3 approximation, contractual EAD, and quarterly macro anchor versus monthly ECL horizon semantics. The `Run ECL` control calls the existing backend only after an explicit button click.

### Model Monitoring

The monitoring page consolidates existing scorecard, PD, LGD, and EAD artifacts. Traffic-light statuses use transparent project monitoring thresholds from `config/monitoring.yaml`; they are portfolio-project thresholds, not regulatory thresholds.

### Report Generator

The Report Generator uses Groq Free Tier by default with `openai/gpt-oss-20b`, configured in `config/reporting.yaml`. It builds a compact `ReportContext` from validated aggregate artifacts, requests structured JSON output, validates the response with Pydantic, checks numerical references against the context, and exposes Markdown, JSON, and HTML downloads. If `GROQ_API_KEY` is missing or Groq is unavailable, the page falls back to deterministic template commentary.

### About / Methodology

The About page summarizes data source, architecture, IFRS 9 methodology, point-in-time controls, model governance, limitations, security boundaries, and the portfolio/research purpose. It explicitly avoids claiming regulatory production readiness.

## Caching And Execution

Artifact loading is cached with `st.cache_data`. Expensive backend executions and Groq calls are guarded by explicit buttons. Scenario, staging, and reporting controls update local UI state first and call backend functions only after the user requests a run.

## Service Boundaries

Page files should only coordinate layout and user controls. Artifact parsing belongs in `app/services/`. Model formulas, scenario mechanics, staging logic, EAD projection, and ECL calculation belong in `src/ifrs9`.

## Groq Setup

Set the key locally in `.env`:

```bash
GROQ_API_KEY=
```

For Streamlit Cloud, configure it only as a secret:

```ini
GROQ_API_KEY="..."
```
