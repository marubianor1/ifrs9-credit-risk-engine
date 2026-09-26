# Freddie Mac Schema Drift

This document records lightweight schema drift checks for the local Freddie Mac Release 47 / July 2026 sample archives in `data/download/`.

## Method

The inspection used Python `zipfile` and did not extract the source data to disk. For each annual ZIP from 2012 through 2026, the check:

- listed ZIP members;
- confirmed expected origination and performance member names;
- sampled the first 100 non-empty records from each member;
- counted pipe-delimited fields;
- checked whether sampled records within each file had consistent field counts.

The machine-readable inventory is stored in `docs/source_schema_inventory.csv`.

## Observed Schema Versions

| Dataset | Schema ID | Years | Observed columns | Evidence |
| --- | --- | --- | ---: | --- |
| Origination | `freddie_origination_v1` | 2012-2026 | 31 | All sampled origination files had 31 pipe-delimited columns and internally consistent sampled rows. |
| Performance | `freddie_performance_v1` | 2012-2026 | 35 | All sampled performance files had 35 pipe-delimited columns and internally consistent sampled rows. |

## Drift Assessment

No physical column-count drift was observed across the local Release 47 annual sample archives from 2012 through 2026.

The evidence supports a single observed annual sample layout for origination and a single observed annual sample layout for performance across the inspected files.

## Resolved Release 47 Layout

The previously unresolved trailing fields are now treated as the Freddie Mac Release 47 / July 2026 disclosure layout for the locally downloaded sample ZIPs.

Origination positions 24-31 are:

| Position | Source field |
| ---: | --- |
| 24 | Seller Name |
| 25 | Super Conforming Flag |
| 26 | Pre-HARP Loan Sequence Number |
| 27 | Special Eligibility Program |
| 28 | HARP Indicator |
| 29 | Property Valuation Method |
| 30 | Interest Only Indicator |
| 31 | VantageScore 4.0 |

Performance positions 31-35 are:

| Position | Source field |
| ---: | --- |
| 31 | Current Period Modification Costs |
| 32 | Current Interest Bearing UPB |
| 33 | Mortgage Insurance Cancellation Indicator |
| 34 | Servicer Name |
| 35 | Bankruptcy Cramdown Costs |

Representative records from 2012, 2017, and 2026 were inspected to confirm that sampled values align with these source field meanings.

## Physical Layout Versus Historical Population

The physical layout is stable in the locally downloaded Release 47 files. That does not mean every field was historically populated.

Some fields may exist as columns in republished historical samples while remaining blank, sentinel-coded, or structurally unavailable for older vintages or older reporting periods. These structural nulls must not be treated as data-quality failures in Bronze or Silver without field-specific availability rules.

## Unified Schema Requirement

A canonical unified schema will still be required later because:

- guide-documented population rules vary by origination date or reporting period;
- some fields exist physically but are only populated for newer loans or periods;
- sentinel codes must remain distinguishable from true nulls;
- structural non-availability must remain distinguishable from ordinary missingness and source sentinel codes.

## Historically Unavailable or Partially Populated Fields

The guide documents the following availability constraints:

- `property_valuation_method`: populated for loans originated on or after January 2017; ACE+ PDR effective for loans originated on or after July 17, 2022.
- `estimated_loan_to_value`: only populated for April 2017 and later periods.
- `delinquency_due_to_disaster`: only populated for January 2014 and later periods.
- `borrower_assistance_status_code`: only populated for January 2014 and later periods.
