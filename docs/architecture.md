# Architecture

The implemented data foundation currently covers source discovery, schema validation, Bronze ingestion, and Silver standardization for Freddie Mac annual sample ZIP files.

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
        -> ingestion manifest and data-quality report
        -> DuckDB validation queries
```

Bronze outputs are written under `data/bronze/freddie/` and Silver outputs are written under `data/silver/freddie/`. Both are ignored by Git. Execution-specific manifests and DQ reports are written under `artifacts/ingestion/` and `artifacts/silver/`, which are also ignored by Git.

## Implemented Components

- `config/schemas/`: versioned source schema registry.
- `src/ifrs9/ingestion/source_inventory.py`: lightweight source ZIP discovery and inspection.
- `src/ifrs9/ingestion/schema_registry.py`: schema metadata loading and validation.
- `src/ifrs9/ingestion/freddie.py`: Freddie Mac Source to Bronze ingestion pipeline.
- `src/ifrs9/transformations/freddie_silver.py`: Freddie Mac Bronze to Silver standardization pipeline.
- `src/ifrs9/transformations/missing_values.py`: field-specific sentinel classification.
- `src/ifrs9/transformations/normalization.py`: date and delinquency normalization helpers.
- `src/ifrs9/transformations/validation.py`: Silver data-quality checks.
- `src/ifrs9/cli.py`: command-line entry point.

## Not Yet Implemented

Default definitions, feature engineering, scorecards, PD, LGD, EAD, SICR, staging, macro overlays, ECL, stress testing, monitoring, and application pages are intentionally out of scope for the current layer.
