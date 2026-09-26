# Data Directory

This project keeps original datasets and generated analytical data out of Git.

Expected local zones:

- `download/`: manually downloaded source ZIP files and vendor documentation.
- `raw/`: immutable local source extracts or controlled raw references.
- `bronze/`: standardized files with minimal processing.
- `silver/`: cleaned and conformed analytical datasets.
- `gold/`: model-ready and reporting-ready datasets.

Do not modify original downloaded files. Reproducible ingestion code will be added in later phases.

