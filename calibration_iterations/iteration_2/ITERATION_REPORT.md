# Iteration 2: Results and Analysis

## Date: 2026-01-23

## Adjustment Applied
- Coverage Ratio: ScaledCohortAvg with 1.2x multiplier (refined from 1.8x in Iteration 1)

## Results

| Metric | Budget | Forecast | Variance | Var % |
|--------|--------|----------|----------|-------|
| Collections | £140.22m | £154.42m | +£14.20m | +10.1% |
| ClosingGBV | £2,404.59m | £2,188.06m | -£216.53m | -9.0% |
| ClosingNBV | £2,006.18m | £1,927.81m | -£78.37m | **-3.9%** |
| Revenue | £51.49m | £47.12m | -£4.36m | -8.5% |
| GrossImpairment | -£28.76m | -£5.83m | +£22.92m | **-79.7%** |
| NetImpairment | -£28.76m | -£5.48m | +£23.28m | **-81.0%** |

## Key Observations

### Unexpected Result
The 1.2x factor produced **less** impairment (£5.83m) than the baseline (£12.14m), which is counter-intuitive.

### Root Cause Analysis
The baseline (Iteration 0) used a mix of approaches:
- `CohortAvg` for mature cohorts (MOB 40+)
- `CohortTrend` for mid-age cohorts (MOB 20-39)
- `DonorCohort` for young cohorts (MOB 0-19)

Iteration 2 changed ALL coverage ratio rules to `ScaledCohortAvg` with 1.2x, which:
1. Lost the `CohortTrend` extrapolation that may have been increasing coverage over time
2. Lost the `DonorCohort` approach that borrowed higher coverage from mature cohorts
3. The 1.2x scaling on a different base approach produced different results

### What This Demonstrates
**Methodology approach matters more than scaling factors.**

The difference between `CohortTrend` (which extrapolates trends) and `ScaledCohortAvg` (which uses historical average × factor) can be more significant than the scaling factor itself.

## NBV Improvement
Despite lower impairment, ClosingNBV improved from:
- Iteration 1 (1.8x): -29.0% variance
- Iteration 2 (1.2x): -3.9% variance ✓

This is because lower provisions = higher NBV, and the NBV is now very close to budget.

## Comparison Across Iterations

| Iteration | Factor | Impairment | NBV | Coverage Approach |
|-----------|--------|------------|-----|-------------------|
| 0 (Baseline) | 1.0x | -£12.14m (-58%) | -3.4% | CohortAvg/Trend/Donor |
| 1 | 1.8x | -£76.04m (+166%) | -29.0% | ScaledCohortAvg |
| 2 | 1.2x | -£5.83m (-80%) | -3.9% | ScaledCohortAvg |
| Budget | - | -£28.76m | - | - |

## Conclusions

1. **NBV is well-matched** - The 1.2x ScaledCohortAvg approach produces NBV within 4% of budget
2. **Impairment remains challenging** - There's a structural difference between model impairment and budget
3. **Approach matters** - Changing from CohortTrend/DonorCohort to ScaledCohortAvg fundamentally changed the model behavior

## Recommendations for Further Work

1. **Preserve original approach logic** - Keep CohortTrend for trending cohorts, DonorCohort for young cohorts
2. **Apply scaling as overlay** - Consider adding a post-calculation scaling overlay instead of changing the approach
3. **Investigate impairment calculation** - The impairment derivation may need review to understand why results are so sensitive to methodology
