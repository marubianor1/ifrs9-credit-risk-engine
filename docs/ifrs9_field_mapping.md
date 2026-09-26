# Preliminary IFRS 9 Field Mapping

This mapping is exploratory. It identifies source fields that may support future IFRS 9 components but does not define final models, thresholds, staging rules, or accounting policy.

Labels:

- `DIRECT`: the field can directly support the component.
- `DERIVED`: the field can support a derived feature or rule.
- `EXTERNAL DATA REQUIRED`: the component needs information not contained in Freddie Mac loan-level files.

## Application and Credit Scoring

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `credit_score`, `vantagescore_4`, `original_debt_to_income_ratio`, `original_loan_to_value`, `original_combined_loan_to_value`, `first_time_homebuyer_flag`, `number_of_borrowers` | Borrower credit quality, affordability, and leverage at origination. |
| DIRECT | `occupancy_status`, `property_type`, `number_of_units`, `loan_purpose`, `channel`, `amortization_type`, `original_loan_term`, `original_interest_rate`, `mortgage_insurance_percentage` | Loan, collateral, and product characteristics. |
| DERIVED | `property_state`, `postal_code`, `metropolitan_statistical_area` | Geographic risk features and joins to local macro or housing data. |

## Behavioural Risk and PD

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `current_loan_delinquency_status`, `loan_age`, `remaining_months_to_legal_maturity`, `current_actual_upb`, `current_interest_rate` | Core monthly performance dynamics. |
| DIRECT | `modification_flag`, `payment_deferral_flag`, `borrower_assistance_status_code`, `delinquency_due_to_disaster` | Behavioural deterioration, hardship, and workout signals. |
| DERIVED | `due_date_of_last_paid_installment`, `period` | Delinquency timing, cure, roll-rate, and vintage features. |

## Default Definition

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `current_loan_delinquency_status` | Supports 90+ DPD rules when status is numeric and at least 3. |
| DIRECT | `zero_balance_code`, `zero_balance_effective_date`, `defect_settlement_date` | Supports absorbing credit-event identification, including short sale, charge-off, third-party sale, REO disposition, and defect events. |
| DIRECT | `current_loan_delinquency_status = RA` | Identifies REO acquisition status. |
| DERIVED | `period`, `due_date_of_last_paid_installment` | Supports default timing and cure window logic. |

## EAD

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `current_actual_upb`, `original_upb`, `current_interest_bearing_upb`, `current_non_interest_bearing_upb`, `zero_balance_removal_upb` | Balance and exposure measures at observation and termination. |
| DIRECT | `current_interest_rate`, `remaining_months_to_legal_maturity`, `original_loan_term` | Required for amortization and exposure projection features. |
| DERIVED | `loan_age`, `period` | Seasoning and time-to-event features. |

## LGD

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `zero_balance_removal_upb`, `mi_recoveries`, `non_mi_recoveries`, `net_sale_proceeds`, `total_expenses`, `legal_costs`, `maintenance_and_preservation_costs`, `taxes_and_insurance`, `miscellaneous_expenses`, `delinquent_accrued_interest`, `actual_loss`, `bankruptcy_cramdown_costs` | Loss, recovery, proceeds, and expense components. |
| DIRECT | `zero_balance_code`, `zero_balance_effective_date` | Disposition type and timing. |
| DERIVED | `period`, `current_loan_delinquency_status` | Time in default, time to liquidation, and status path features. |

## SICR and Staging

| Label | Fields | Rationale |
| --- | --- | --- |
| DIRECT | `current_loan_delinquency_status`, `modification_flag`, `payment_deferral_flag`, `borrower_assistance_status_code` | Observable credit deterioration and assistance/workout signals. |
| DERIVED | `credit_score`, `original_debt_to_income_ratio`, `original_loan_to_value`, `original_combined_loan_to_value`, `current_actual_upb`, `loan_age` | Relative risk deterioration and segmentation features may be derived later. |
| EXTERNAL DATA REQUIRED | Lifetime PD threshold calibration and macro overlays | IFRS 9 SICR assessment requires policy choices and model outputs beyond raw Freddie fields. |

## Forward-Looking and ECL

| Label | Fields | Rationale |
| --- | --- | --- |
| DERIVED | `property_state`, `postal_code`, `metropolitan_statistical_area`, `period`, `first_payment_date` | Keys for joining macroeconomic, unemployment, home price, and vintage data. |
| DIRECT | `current_actual_upb`, `current_interest_rate`, `remaining_months_to_legal_maturity`, `current_interest_bearing_upb`, `current_non_interest_bearing_upb` | Exposure and discounting inputs. |
| EXTERNAL DATA REQUIRED | Unemployment, rates, house price indexes, macro scenarios, scenario weights | These are not contained in the Freddie Mac loan-level files and must not be fabricated. |
