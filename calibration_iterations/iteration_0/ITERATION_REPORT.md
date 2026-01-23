# Iteration 0: Baseline Analysis

## Date: 2026-01-23 17:33

## Purpose
This is the baseline iteration using the **original methodology** with proper modeling logic:
- `CohortAvg` for stable/mature cohorts
- `CohortTrend` for cohorts still seasoning
- `DonorCohort` for young cohorts borrowing from mature ones

No scaling factors or budget-matching adjustments are applied.

## Methodology Summary
The original methodology uses:
- **Collections/Revenue**: 6-period rolling average (`CohortAvg`)
- **Coverage Ratios**:
  - MOB 40+: `CohortAvg` (rates have stabilised)
  - MOB 20-39: `CohortTrend` (still trending)
  - MOB 0-19: `DonorCohort` (borrow from mature cohorts)

## Variance Analysis

This baseline shows where the pure model differs from budget, helping us understand:
1. Whether the budget assumptions are different from historical actuals
2. Which metrics need adjustment and why
3. The magnitude of adjustments needed

## Key Observations

Based on the variance analysis, we observe:

- **Collections**: Forecast is +10.1% vs budget
- **Revenue**: Forecast is -8.5% vs budget
- **GrossImpairment**: Forecast is -57.8% vs budget
- **ClosingGBV**: Forecast is -9.0% vs budget

## Next Steps
Analyze the variances to determine appropriate modeling-based adjustments.
