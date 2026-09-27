# IFRS 9 Credit Risk Engine

Professional portfolio project to build a reproducible IFRS 9 credit risk and expected credit loss framework in Python.

The intended end-to-end architecture covers data ingestion, data quality, default definition, scoring, rating, 12-month and lifetime PD, EAD, LGD, SICR, staging, forward-looking macroeconomic adjustment, scenario weighting, ECL, stress testing, validation, monitoring, reporting, and an application layer.

Current status: **Modelling — Calibrated PD Framework**.

The Source to Bronze ingestion pipeline, Bronze to Silver standardization pipeline, Gold point-in-time loan-month mart, default definition, PD target factory, temporal development-sample factory, traditional logistic scorecard experiment framework, and calibrated PD framework have been implemented for the local Freddie Mac sample archives. No LGD, EAD, SICR, staging, macro overlay, or ECL estimates have been implemented yet.

The current project layer contains a lightweight Freddie Mac source inventory, schema drift notes, version-controlled Release 47 source and Silver schemas, a reproducible Bronze Parquet ingestion pipeline, a Silver standardization pipeline, a Gold analytical mart with leakage-aware feature registry selectors, compact default/PD target tables, temporal development metadata, logistic WOE scorecard runs for application and behavioural populations, and calibrated behavioural PD artifacts with rating, TTC, lifetime, transition, and backtesting outputs.

## Dataset

The primary dataset planned for this project is the Freddie Mac Single-Family Loan-Level Dataset. Original datasets are not included in this repository and must be obtained from the official source under the applicable terms of use.

Downloaded source files should remain outside version control. This project uses the following data-zone convention:

- `data/raw/`: immutable source extracts or controlled local references.
- `data/bronze/`: standardized but minimally processed data.
- `data/silver/`: cleaned and conformed analytical data.
- `data/gold/`: model-ready and reporting-ready datasets.

## Reproducibility

The project is configured with Poetry for dependency management. Configuration is stored under `config/` using repository-relative paths so the project can be reproduced on another machine.

## Planned Components

- ETL and ingestion for loan-level origination and monthly performance files.
- Data quality checks and audit reporting.
- Default definition and target construction.
- Credit scoring and rating calibration.
- PD, LGD, and EAD modelling.
- SICR rules and IFRS 9 staging.
- Forward-looking macroeconomic scenarios and Vasicek-style adjustments.
- ECL calculation, stress testing, validation, monitoring, and reporting.
- Streamlit application for exploration and final presentation.

## Installation

Install Poetry, then from the repository root run:

```bash
poetry install
```

Run Freddie Mac Source to Bronze ingestion for one year:

```bash
poetry run ifrs9 ingest-freddie --year 2012
```

Run all discovered local sample years:

```bash
poetry run ifrs9 ingest-freddie --all
```

Build the Silver layer:

```bash
poetry run ifrs9 build-silver --all
```

Build the Gold point-in-time mart:

```bash
poetry run ifrs9 build-mart --all
```

Build default events and PD target labels:

```bash
poetry run ifrs9 build-targets --all
```

Build temporal development sample metadata:

```bash
poetry run ifrs9 build-development-sample
```

Train logistic scorecards:

```bash
poetry run ifrs9 train-scorecard --population behavioural
poetry run ifrs9 train-scorecard --population application
```

Build calibrated PD artifacts from an existing scorecard run:

```bash
poetry run ifrs9 build-pd --scorecard-run behavioural_qe_v1
```

Run quality checks after the environment is installed:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```
