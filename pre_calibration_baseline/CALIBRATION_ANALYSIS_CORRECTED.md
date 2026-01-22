# Calibration Analysis - Corrected Findings

**Date:** 2026-01-22
**Analysis Type:** Backbook Model vs Backbook Budget Comparison

---

## Executive Summary

After correcting the comparison methodology, the actual variances are:

| Metric | Model vs Budget | Direction |
|--------|-----------------|-----------|
| **Total Collections** | +18.5% | Model collects MORE |
| **Closing GBV (Sep 2026)** | -19.9% | Model runs off FASTER |

This is internally consistent: higher collections → faster runoff → lower ending balances.

---

## Monthly Variance Analysis

### Collections (Total Portfolio)

| Month | Model | Budget | Variance |
|-------|-------|--------|----------|
| Nov 2025 | £12.8M | £16.2M | -20.9% |
| Dec 2025 | £13.2M | £15.5M | -14.9% |
| Jan 2026 | £13.1M | £14.7M | -10.8% |
| **Feb 2026** | **£17.6M** | **£13.8M** | **+27.5%** |
| **Mar 2026** | **£17.5M** | **£13.1M** | **+33.1%** |
| Apr 2026 | £16.2M | £12.5M | +29.2% |
| May 2026 | £15.3M | £11.9M | +28.3% |
| Jun 2026 | £14.0M | £11.4M | +23.2% |
| Jul 2026 | £12.6M | £10.8M | +16.0% |
| Aug 2026 | £11.6M | £10.3M | +12.0% |
| Sep 2026 | £10.7M | £10.0M | +7.0% |

**Key Observation:** The model shows a significant spike in Feb-Mar 2026, then gradually converges toward budget by Sep 2026.

### Closing GBV (Total Portfolio)

| Month | Model | Budget | Variance |
|-------|-------|--------|----------|
| Nov 2025 | £258.5M | £255.0M | +1.4% |
| Dec 2025 | £244.9M | £251.3M | -2.5% |
| Jan 2026 | £236.0M | £243.9M | -3.2% |
| Feb 2026 | £223.6M | £235.1M | -4.9% |
| Mar 2026 | £207.2M | £226.5M | -8.5% |
| Apr 2026 | £195.8M | £217.3M | -9.9% |
| May 2026 | £185.0M | £209.0M | -11.5% |
| Jun 2026 | £172.0M | £202.2M | -14.9% |
| Jul 2026 | £163.4M | £194.2M | -15.8% |
| Aug 2026 | £155.6M | £188.0M | -17.2% |
| Sep 2026 | £145.8M | £182.0M | -19.9% |

---

## Segment-Level Analysis

### Collections Variance by Segment (12-month aligned total)

| Segment | Model | Budget | Variance |
|---------|-------|--------|----------|
| Non Prime | £83.6M | £74.4M | **+12.4%** |
| Near Prime Small | £27.4M | £23.0M | **+19.0%** |
| Near Prime Medium | £52.3M | £40.9M | **+27.9%** |
| Prime | £2.9M | £1.9M | **+52.8%** |
| **Total** | **£166.2M** | **£140.2M** | **+18.5%** |

### Key Findings by Segment

1. **Non Prime (+12.4%)**: Collecting slightly faster than budget
2. **Near Prime Small (+19.0%)**: Collecting noticeably faster than budget
3. **Near Prime Medium (+27.9%)**: Collecting significantly faster than budget
4. **Prime (+52.8%)**: Collecting much faster than budget - this is the largest variance

---

## Business Logic for Required Adjustments

### Why the Model Collects More Than Budget

The model uses **CohortAvg** approach for collections with a 6-month lookback window. This means:
- Collection rates are based on recent historical performance
- If recent months had higher-than-average collections, the model projects this forward
- The Feb-Mar spike suggests seasonal factors may be over-amplifying certain months

### Recommended Calibration Actions

To align the model with budget, we would need to **reduce** collection rates:

| Segment | Current Approach | Suggested Adjustment |
|---------|------------------|---------------------|
| Non Prime | CohortAvg(6) | Reduce by ~10-15% or use longer averaging window |
| Near Prime Small | CohortAvg(6) | Reduce by ~15-20% or use longer averaging window |
| Near Prime Medium | CohortAvg(6) | Reduce by ~20-25% or use longer averaging window |
| Prime | CohortAvg(6) | Reduce by ~35-40% or use longer averaging window |

### Business Rationale for Adjustments

1. **Longer averaging window (e.g., 12 months instead of 6)**:
   - Smooths out seasonal volatility
   - Reduces impact of recent high-collection months
   - More conservative projection

2. **Apply a scaling factor < 1.0**:
   - If budget assumes macro headwinds (economic slowdown, affordability pressures)
   - Model may be too optimistic based on historical performance

3. **Review seasonal factors**:
   - The Feb-Mar spike is suspicious - check if seasonal adjustments are too aggressive
   - Budget may use different seasonality assumptions

---

## Important Caveats

### What This Analysis Assumes

1. Budget is the "truth" we should calibrate toward
2. Budget and model use same segment definitions
3. Budget timing (Nov 2025 start) is correctly aligned with our Oct 2025 start

### What This Analysis Does NOT Address

1. Whether budget collection rates are realistic
2. Whether historical data supports budget assumptions
3. Impairment and coverage ratio alignment (separate analysis needed)

---

## Files Updated

- Original methodology preserved in: `Rate_Methodology_ORIGINAL.csv`
- This analysis: `CALIBRATION_ANALYSIS_CORRECTED.md`
- Comparison data: `Original_Forecast_vs_Budget.csv`

---

## Next Steps

1. **Validate seasonal factors**: Check Feb-Mar amplification
2. **Test longer averaging window**: Try CohortAvg(12) instead of CohortAvg(6)
3. **Apply segment-specific scaling**: Reduce collection rates to match budget
4. **Re-run impairment analysis**: After calibrating collections, reassess coverage ratios
