# Bronze Layer

The Bronze layer is the first durable analytical representation of the Freddie Mac source sample files. It is source-faithful and intended to support later Silver normalization without changing source meaning prematurely.

## Source

Bronze ingestion reads `sample_YEAR.zip` archives from the configured Freddie Mac source directory. It reads `sample_orig_YEAR.txt` and `sample_perf_YEAR.txt` directly from each ZIP file and does not permanently extract TXT files to disk.

## Schema Enforcement

Each source member is validated against the version-controlled schema registry in `config/schemas/`.

The ingestion policy is fail-fast for structural schema errors:

- expected column count must match exactly;
- columns are mapped by position;
- malformed rows are rejected before Parquet is written;
- columns are never silently shifted.

## Partitioning

Bronze files are written under:

```text
data/bronze/freddie/
  origination/vintage_year=YYYY/part-origination.parquet
  performance/vintage_year=YYYY/part-performance.parquet
```

One Parquet file per dataset/year is used for the current sample scale.

## Type Policy

Bronze applies conservative schema-driven typing:

- identifiers, postal codes, categories, and ambiguous fields remain strings;
- monetary, balance, rate, ratio, and count fields are cast to numeric types where safe;
- `YYYYMM` period/date fields remain strings to preserve source representation exactly;
- fields with mixed numeric/code semantics remain strings.

## Missing and Sentinel Policy

Bronze preserves Freddie Mac sentinel codes such as `9999`, `999`, `99`, `9`, `7`, `U`, `RA`, and `XX` according to the source values.

Blank fields are represented as empty strings for string columns and nulls after safe numeric casts for numeric columns. Field-specific normalization is deferred to Silver so that actual nulls, sentinel codes, not-applicable values, unknown values, and structural non-availability can be treated deliberately.

## Lineage Columns

Bronze appends technical metadata columns prefixed with `_`:

- `_source_year`
- `_source_zip`
- `_source_member`
- `_ingested_at`
- `_schema_version`
- `_source_row_number`

`_source_row_number` is assigned after reading each source member and reflects the non-empty source row order within that member.

## Rerun Behavior

Without `--force`, an existing successful partition is skipped. With `--force`, only the requested year partitions are regenerated.

Writes use a temporary partition directory followed by a rename into the final location. If ingestion fails before completion, the temporary output is removed and existing valid Bronze data is preserved.

## Data-Quality Controls

The pipeline reports, at minimum:

- row counts;
- column counts;
- unique loan counts;
- duplicate origination loan IDs;
- duplicate performance `(loan_id, period)` records;
- null loan IDs and periods;
- performance min/max periods;
- same-year referential integrity between performance and origination;
- source ZIP size, Parquet size, and processing time.

Generated outputs:

- `artifacts/ingestion/bronze_manifest.json`
- `artifacts/ingestion/bronze_data_quality.csv`
- `artifacts/ingestion/bronze_summary.md`

## Known Limitations

Bronze does not impute, derive default, derive SICR, create modelling features, join macroeconomic data, or calculate IFRS 9 measures.

The Release 47 physical layout is stable in the local files, but historical field population can vary. Structural non-availability should not be interpreted as a data-quality error without field-specific rules.
