# IFRS 9 Credit Risk Engine

Professional portfolio project to build a reproducible IFRS 9 credit risk and expected credit loss framework in Python.

The intended end-to-end architecture covers data ingestion, data quality, default definition, scoring, rating, 12-month and lifetime PD, EAD, LGD, SICR, staging, forward-looking macroeconomic adjustment, scenario weighting, ECL, stress testing, validation, monitoring, reporting, and an application layer.

Current status: **Data foundation — source schema analysis**.

No ETL pipeline, models, transformations, or IFRS 9 estimates have been implemented yet.

The current project layer contains a lightweight Freddie Mac source inventory, schema drift notes, and a version-controlled schema registry for the observed sample files.

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

Run quality checks after the environment is installed:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```
