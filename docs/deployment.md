# Public Deployment

This project supports two execution modes:

- `local_full`: full local research workflow with training, recalibration, rebuilds, and full artifacts.
- `cloud_demo`: public Streamlit demo using only compact aggregate artifacts under `deployment/`.

Set the mode with:

```bash
IFRS9_APP_MODE=cloud_demo
```

For Streamlit Community Cloud, set `IFRS9_APP_MODE="cloud_demo"` in app secrets.

## Build Bundle

Create the public deployment bundle:

```bash
poetry run ifrs9 build-deployment-bundle
```

The bundle is deterministic and contains:

```text
deployment/
├── manifest.json
├── portfolio/
├── scoring/
├── pd/
├── lgd/
├── ead/
├── sicr/
├── ecl/
├── scenarios/
└── monitoring/
```

Only aggregate, non-reconstructable research outputs are included. Raw Freddie Mac files,
Bronze/Silver/Gold loan-level data, loan-level targets, staging, ECL snapshots, scored rows,
and loan-level predictions are excluded.

## Safety Audit

The deployment builder runs an automated safety audit and fails if it detects:

- prohibited loan-level or raw-data file patterns;
- absolute local paths such as `/Users/...`;
- secret-looking values such as Groq API keys.

The audit is also covered by tests.

## Cloud Behavior

In `cloud_demo`, the app enables aggregate dashboards, preset Scenario Lab comparisons, model
monitoring, and report generation. Full recalculation buttons are hidden or replaced with:

```text
Full recalculation is available in local/full mode.
```

Custom Scenario Lab execution is disabled in cloud mode. The page compares only the packaged
Baseline, Mild deterioration, and Severe deterioration aggregate results.

## Groq Reporting

The Report Generator uses Groq only when both conditions are true:

- `GROQ_API_KEY` is configured in environment or Streamlit secrets;
- `ENABLE_LLM_REPORTING=true`.

If the key is missing, the provider is unavailable, or rate limits occur, the app uses a
deterministic template fallback.

## Streamlit Cloud Setup

Use Python 3.11 in Streamlit Cloud Advanced Settings.

Add secrets only through Streamlit Cloud:

```toml
GROQ_API_KEY="..."
ENABLE_LLM_REPORTING=true
IFRS9_APP_MODE="cloud_demo"
```

Do not commit `.streamlit/secrets.toml`.
