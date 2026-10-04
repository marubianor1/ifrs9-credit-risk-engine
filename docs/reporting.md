# AI-assisted Reporting

The report generator creates AI-assisted narrative commentary from validated IFRS 9 artifacts. It is a reporting layer only: it does not calculate, modify, refit, or overwrite PD, LGD, EAD, SICR, Stage, ECL, scenario results, or model metrics.

## Provider

The default provider is Groq Free Tier using:

```yaml
provider: groq
model: openai/gpt-oss-20b
reasoning_effort: low
max_output_tokens: 4000
```

Provider and model settings live in `config/reporting.yaml`.

## Secret Configuration

For local development, copy `.env.example` to `.env` and set:

```bash
GROQ_API_KEY=
```

For Streamlit Cloud, configure the key as a secret:

```ini
GROQ_API_KEY="..."
```

Do not commit `.env` or `.streamlit/secrets.toml`.

## Governance Boundary

The reporting flow is:

```text
validated artifacts
-> ReportContext
-> Groq LLM
-> structured narrative
```

`ReportContext` contains compact aggregate metrics, source run IDs, methodology notes, and limitations. It does not include Freddie Mac source files, Gold loan-month data, loan-level ECL output, or other loan-level payloads.

## Structured Output

The Groq call requests strict JSON schema output matching the Pydantic `GeneratedReport` model. Sections include:

- `executive_summary`
- `portfolio_position`
- `key_risk_movements`
- `model_performance`
- `scenario_analysis`
- `limitations`
- `management_actions`

If the provider is unavailable, the app uses a deterministic template-based report.

## Numerical Validation

After generation, the reporting layer extracts material numerical references from the narrative and checks them against number variants derived from `ReportContext`. Unsupported numerical claims are surfaced in the UI and exports. They are not silently accepted.

## Exports

The app supports Markdown, JSON, and lightweight HTML downloads. PDF generation is intentionally out of scope for this phase.

## Limitations

This is a portfolio/research implementation, not a regulatory production system. Generated narrative is an aid for review and presentation, not a substitute for model owner approval, independent validation, governance sign-off, or regulatory reporting controls.
