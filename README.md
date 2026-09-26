# IFRS 9 Credit Risk Engine

Professional portfolio project to build a reproducible IFRS 9 credit risk and expected credit loss framework in Python.

The intended end-to-end architecture covers data ingestion, data quality, default definition, scoring, rating, 12-month and lifetime PD, EAD, LGD, SICR, staging, forward-looking macroeconomic adjustment, scenario weighting, ECL, stress testing, validation, monitoring, reporting, and an application layer.

Current status: **Data Foundation — Silver standardized layer**.

The Source to Bronze ingestion pipeline and Bronze to Silver standardization pipeline have been implemented for the local Freddie Mac sample archives. No modelling features, default definitions, PD, LGD, EAD, SICR, staging, or IFRS 9 estimates have been implemented yet.

The current project layer contains a lightweight Freddie Mac source inventory, schema drift notes, version-controlled Release 47 source and Silver schemas, a reproducible Bronze Parquet ingestion pipeline, and a Silver standardization pipeline.

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

Run quality checks after the environment is installed:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```
