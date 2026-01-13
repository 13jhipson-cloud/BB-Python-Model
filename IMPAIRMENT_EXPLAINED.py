#!/usr/bin/env python3
"""
IMPAIRMENT CALCULATION EXPLAINED

This script provides a detailed walkthrough of how impairment is calculated
in the backbook forecast model, with a step-by-step example.
"""

print("""
================================================================================
IMPAIRMENT CALCULATION - COMPLETE EXPLANATION
================================================================================

The model calculates several impairment metrics. Here's what each means:

--------------------------------------------------------------------------------
KEY CONCEPTS
--------------------------------------------------------------------------------

1. PROVISION BALANCE (aka Allowance for Credit Losses)
   - The reserve set aside to cover expected future losses
   - Calculated as: ClosingGBV × Coverage_Ratio
   - Example: £100,000 GBV × 12% coverage = £12,000 provision

2. COVERAGE RATIO
   - What % of GBV is covered by provisions
   - Higher = more conservative/risky portfolio
   - Typical range: 5% (prime) to 50%+ (subprime)

3. PROVISION MOVEMENT
   - Change in provision balance from prior month
   - Positive = increased provisions (expense)
   - Negative = released provisions (income)

4. DEBT SALE (when you sell bad loans to a third party)
   - Debt_Sale_WriteOffs: GBV of loans sold
   - Debt_Sale_Provision_Release: Provision freed up (was covering those loans)
   - Debt_Sale_Proceeds: Cash received from buyer

--------------------------------------------------------------------------------
THE IMPAIRMENT FORMULAS
--------------------------------------------------------------------------------

STEP 1: Calculate Provision Balance
    Provision_Balance = ClosingGBV × Coverage_Ratio

STEP 2: Calculate Provision Movement
    Provision_Movement = Provision_Balance[t] - Provision_Balance[t-1]

STEP 3: If Debt Sale occurs this month:
    Debt_Sale_Provision_Release = Debt_Sale_WriteOffs × Debt_Sale_Coverage_Ratio
    Debt_Sale_Proceeds = Debt_Sale_WriteOffs × Debt_Sale_Proceeds_Rate

STEP 4: Calculate Net Impairment components
    Non_DS_Provision_Movement = Provision_Movement + Debt_Sale_Provision_Release
    Gross_Impairment_ExcludingDS = Non_DS_Provision_Movement + WO_Other
    Debt_Sale_Impact = Debt_Sale_WriteOffs + Debt_Sale_Provision_Release + Debt_Sale_Proceeds

STEP 5: Total Net Impairment
    Net_Impairment = Gross_Impairment_ExcludingDS + Debt_Sale_Impact

STEP 6: Closing NBV
    ClosingNBV = ClosingGBV - Net_Impairment

================================================================================
END-TO-END EXAMPLE: Forecasting Month 1
================================================================================

Starting Point (from last actual month - September 2025):
    OpeningGBV = £98,748.52
    Prior_Provision_Balance = £98,217.87
    Coverage_Ratio (from methodology) = 12.5% (Manual setting)

STEP 1: Calculate ClosingGBV (using collection rates)
    ClosingGBV = OpeningGBV + InterestRevenue - Collections - WriteOffs
    ClosingGBV = £98,748.52 + £822.90 - £0 - £0 - £0 - £0
    ClosingGBV = £99,571.42

STEP 2: Calculate New Provision Balance
    Provision_Balance = ClosingGBV × Coverage_Ratio
    Provision_Balance = £99,571.42 × 0.125
    Provision_Balance = £12,446.43

STEP 3: Calculate Provision Movement
    Provision_Movement = New Balance - Prior Balance
    Provision_Movement = £12,446.43 - £98,217.87
    Provision_Movement = -£85,771.44  (RELEASE - prior provision was much higher!)

STEP 4: No Debt Sale this month
    Debt_Sale_WriteOffs = £0
    Debt_Sale_Provision_Release = £0
    Debt_Sale_Proceeds = £0

STEP 5: Calculate Net Impairment
    Non_DS_Provision_Movement = -£85,771.44 + £0 = -£85,771.44
    Gross_Impairment_ExcludingDS = -£85,771.44 + £0 = -£85,771.44
    Debt_Sale_Impact = £0 + £0 + £0 = £0
    Net_Impairment = -£85,771.44 + £0 = -£85,771.44

STEP 6: Calculate Closing NBV
    ClosingNBV = ClosingGBV - Net_Impairment
    ClosingNBV = £99,571.42 - (-£85,771.44)
    ClosingNBV = £185,342.87

NOTE: The large negative impairment (provision release) occurs because:
- The actual coverage ratio in the data was ~99.5% (very high provisions)
- The forecast methodology uses 12.5% coverage ratio
- This difference causes a big provision release in month 1

================================================================================
DEBT SALE EXAMPLE
================================================================================

If a Debt Sale Schedule is provided with:
    Debt_Sale_WriteOffs = £10,000
    Debt_Sale_Coverage_Ratio = 85%
    Debt_Sale_Proceeds_Rate = 90%

Then:
    Debt_Sale_Provision_Release = £10,000 × 0.85 = £8,500 (provision freed up)
    Debt_Sale_Proceeds = £10,000 × 0.90 = £9,000 (cash received)

    Debt_Sale_Impact = £10,000 + £8,500 + £9,000 = £27,500

This represents:
    - £10,000 of bad loans removed from GBV
    - £8,500 of provisions released (no longer needed)
    - £9,000 cash received from the buyer

================================================================================
WHAT THE DEBT_SALE_SCHEDULE FILE DOES
================================================================================

The Debt_Sale_Schedule.csv allows you to specify PLANNED debt sales:

    ForecastMonth, Segment, Cohort, Debt_Sale_WriteOffs, Debt_Sale_Coverage_Ratio, Debt_Sale_Proceeds_Rate
    6/30/2025, NRP-S, 202001, 5000.00, 0.85, 0.90

This tells the model:
- In June 2025, for NRP-S/202001 cohort
- Sell £5,000 of bad loans
- Release 85% of provisions on those loans
- Receive 90% of face value as proceeds

If you DON'T provide this file, the model assumes NO debt sales occur
during the forecast period.

================================================================================
""")
