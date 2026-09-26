# Architecture

The implemented analytical foundation currently covers source discovery, schema validation, Bronze ingestion, Silver standardization, and a Gold point-in-time loan-month mart for Freddie Mac annual sample ZIP files.

## Current Flow

```text
Freddie Mac sample ZIP files
        -> source discovery
        -> schema registry validation
        -> positional column mapping
        -> safe type casting
        -> Bronze Parquet partitions
        -> field-specific normalization
        -> monthly date standardization
        -> Silver Parquet partitions
        -> Gold point-in-time loan-static and loan-month partitions
        -> leakage-aware feature registry
        -> Gold DuckDB analytical views
        -> ingestion manifest and data-quality report
        -> DuckDB validation queries
```

Bronze outputs are written under `data/bronze/freddie/`, Silver outputs are written under `data/silver/freddie/`, and Gold outputs are written under `data/gold/freddie/`. These data outputs are ignored by Git. Execution-specific manifests and DQ reports are written under `artifacts/ingestion/`, `artifacts/silver/`, and `artifacts/mart/`, which are also ignored by Git.

## Implemented Components

- `config/schemas/`: versioned source schema registry.
- `src/ifrs9/ingestion/source_inventory.py`: lightweight source ZIP discovery and inspection.
- `src/ifrs9/ingestion/schema_registry.py`: schema metadata loading and validation.
- `src/ifrs9/ingestion/freddie.py`: Freddie Mac Source to Bronze ingestion pipeline.
- `src/ifrs9/transformations/freddie_silver.py`: Freddie Mac Bronze to Silver standardization pipeline.
- `src/ifrs9/transformations/missing_values.py`: field-specific sentinel classification.
- `src/ifrs9/transformations/normalization.py`: date and delinquency normalization helpers.
- `src/ifrs9/transformations/validation.py`: Silver data-quality checks.
- `src/ifrs9/mart/loan_static.py`: Gold loan-static table construction.
- `src/ifrs9/mart/loan_month.py`: Gold point-in-time loan-month mart construction and artifacts.
- `src/ifrs9/mart/feature_registry.py`: leakage-aware feature registry loader and selectors.
- `src/ifrs9/mart/views.py`: DuckDB Gold view registration.
- `src/ifrs9/mart/validation.py`: Gold mart validation checks.
- `src/ifrs9/cli.py`: command-line entry point.

## Not Yet Implemented

Default definitions, target construction, scorecards, PD, LGD, EAD, SICR, staging, macro overlays, ECL, stress testing, monitoring, and application pages are intentionally out of scope for the current layer.
