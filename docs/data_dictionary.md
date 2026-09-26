# Data Dictionary

This is the initial project-level overview of the Freddie Mac Single-Family Loan-Level Dataset sample files. It is based on the Freddie Mac Release 47 / July 2026 disclosure layout, the local user guide, and lightweight inspection of the annual sample ZIP files.

## Source

Primary source data is the Freddie Mac Single-Family Loan-Level Dataset sample, stored locally under `data/download/`. Original source data is ignored by Git and is not redistributed in this repository.

Local guide:

- `data/download/dictionary/user_guide.pdf`

## Files

Each annual sample ZIP from 2012 through 2026 contains:

- `sample_orig_YEAR.txt`: headerless loan-level origination records.
- `sample_perf_YEAR.txt`: headerless monthly loan performance records.

Both files use pipe (`|`) delimiters.

## Observation Unit

Origination files contain one record per loan at acquisition/origination disclosure.

Performance files contain multiple monthly records per loan from Freddie Mac acquisition through the earlier of termination event or performance cutoff date.

## Major Variable Groups

- Identifiers: loan sequence number and refinance link fields.
- Borrower and underwriting: credit score, DTI, first-time homebuyer flag, number of borrowers.
- Collateral: property state, postal code, MSA, property type, occupancy, units, valuation method.
- Origination terms: original UPB, LTV, CLTV, interest rate, channel, loan purpose, term, amortization type.
- Servicing and balances: current actual UPB, interest-bearing UPB, non-interest-bearing UPB, current interest rate, loan age, remaining maturity.
- Delinquency and default: delinquency status, DDLPI, zero-balance code, zero-balance effective date, defect settlement date, REO acquisition status.
- Modification and assistance: modification flag, step indicator, payment deferral flag, modification costs, borrower assistance status.
- Recovery and loss: MI recoveries, non-MI recoveries, net sale proceeds, expenses, zero-balance removal UPB, delinquent accrued interest, actual loss calculation.

## Missing and Sentinel Conventions

The project must not collapse all missing-like values into null at this stage. Documented and observed conventions include:

- blank/null: often means not applicable, not modified, no workout plan, or not populated depending on the field;
- `9999`: credit score or VantageScore 4.0 not available, depending on field;
- `999`: not available or unknown for fields such as MI percentage, LTV, CLTV, and estimated LTV depending on context;
- `99`: not available for fields such as number of units, property type, and number of borrowers;
- `9`: not available, not applicable, or not disclosed for selected categorical flags;
- `7`: not available/not applicable for selected newer fields;
- `U`: unknown for net sale proceeds;
- `RA`: REO acquisition in delinquency status;
- `XX`: observed in initial performance rows and preserved as a source delinquency-status value pending downstream business interpretation.

## Data Limitations

The dataset is a disclosure dataset, not a servicing system of record. The guide notes possible timing mismatches between accounting, default reporting, and termination event cycles. Dates are masked to month granularity, postal codes are masked, original UPB is rounded, early current UPB may be rounded, and seller/servicer names may be grouped.

## Schema Versions

The observed local Release 47 annual samples use:

- `freddie_origination_v1`: 31 columns for years 2012-2026.
- `freddie_performance_v1`: 35 columns for years 2012-2026.

Resolved Release 47 tail fields:

- Origination positions 24-31: Seller Name, Super Conforming Flag, Pre-HARP Loan Sequence Number, Special Eligibility Program, HARP Indicator, Property Valuation Method, Interest Only Indicator, VantageScore 4.0.
- Performance positions 31-35: Current Period Modification Costs, Current Interest Bearing UPB, Mortgage Insurance Cancellation Indicator, Servicer Name, Bankruptcy Cramdown Costs.

The physical layout is stable in the local Release 47 files, but historical field availability and population can vary. A blank or sentinel value in an old period may represent structural non-availability rather than poor source quality.

See `docs/schema_drift.md` and `config/schemas/` for details.

## Preliminary IFRS 9 Relevance

This mapping is preliminary and does not define final methodology.

- Scoring: credit score, DTI, LTV, CLTV, occupancy, property type, loan purpose, channel, first-time homebuyer flag, number of borrowers, MI percentage, loan term, geography.
- PD: monthly delinquency status, loan age, remaining maturity, current balance, current rate, modification flags, assistance status, disaster flag, origination risk factors.
- EAD: current actual UPB, original UPB, current interest-bearing UPB, current non-interest-bearing UPB, current interest rate, remaining maturity, zero-balance removal UPB.
- LGD: zero-balance code, zero-balance removal UPB, MI recoveries, non-MI recoveries, net sale proceeds, total and component expenses, delinquent accrued interest, actual loss.
- SICR: delinquency migration, modification status, payment deferral, borrower assistance, balance behavior, loan age, remaining maturity.
- Staging: delinquency status, 30+ DPD and 90+ DPD derivations, modification/payment assistance signals, default/termination events.
- Forward-looking and ECL: property geography, observation dates, balances, rates, maturity, credit quality and collateral variables; external macroeconomic data is required for macro overlays.
