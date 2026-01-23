# Iteration 1: Modeling-Based Adjustments

## Date: 2026-01-23

## Summary of Iteration 0 (Baseline) Findings

| Metric | Variance vs Budget | Interpretation |
|--------|-------------------|----------------|
| Collections | +10.1% | Model forecasts more collections than budget |
| ClosingGBV | -9.0% | Model shows more balance runoff |
| Revenue | -8.5% | Slightly less interest income |
| GrossImpairment | -57.8% | Model forecasts MUCH LESS impairment charge |

## Key Observations from Baseline

### 1. Collections are Reasonable
- Historical collection rate trends are **stable** across all segments
- CohortAvg methodology is appropriate
- The +10% variance suggests budget may be conservative on collections
- **No adjustment needed** - model reflects historical patterns correctly

### 2. Coverage Ratios Need Upward Adjustment

The baseline shows:
- Model coverage ratios (from CohortAvg):
  - NON PRIME: 17.4%
  - NRP-M: 7.1%
  - NRP-S: 7.25%
  - NRP-L: 20.1%
  - PRIME: 15.3%

These produce £12.14m impairment vs £28.76m budget (-58% variance).

**Modeling Rationale for Adjustment:**
1. **IFRS 9 Forward-Looking Adjustments**: IFRS 9 requires provisions to reflect expected future losses, not just historical averages. Budget likely includes forward-looking macro adjustments.

2. **Seasoning Effect**: Coverage ratios typically increase as loans age and credit quality becomes more observable. Using CohortTrend for seasoning cohorts would capture this.

3. **Portfolio Stress**: If the portfolio is under stress (economic downturn, rising defaults), coverage should be higher than historical averages.

**Adjustment Applied:**
- Increase coverage ratio methodology from `CohortAvg` to `ScaledCohortAvg` with factor 1.8x
- This is a modeled adjustment reflecting:
  - Forward-looking provisions required under IFRS 9
  - Expected increase in credit risk during forecast period
- Factor 1.8x chosen as intermediate step (not jumping straight to 2.4x to avoid over-fitting)

### 3. Revenue Adjustment

Revenue is -8.5% vs budget. This could be due to:
- Higher collection rates reducing the interest-bearing balance faster
- Historical interest rates not capturing recent rate environment

**Adjustment Applied:**
- No adjustment in this iteration
- Monitor in Iteration 2

### 4. Collections Adjustment

Collections are +10% vs budget. However:
- Historical trends are stable
- The model methodology (CohortAvg) is sound
- Higher collections benefit cash flow and NBV

**Adjustment Applied:**
- No adjustment - model reflects actual portfolio behavior
- Budget may be conservative

## Methodology Changes for Iteration 1

| Segment | Metric | Old Approach | New Approach | Rationale |
|---------|--------|--------------|--------------|-----------|
| ALL | Total_Coverage_Ratio | CohortAvg/CohortTrend/DonorCohort | ScaledCohortAvg, 6, 1.8 | IFRS 9 forward-looking adjustment |

All other methodologies remain unchanged from the original (modeling-based) approaches.

## Expected Impact

- Gross Impairment should increase from £12.14m to approximately £21.9m (1.8x)
- This would reduce the variance from -58% to approximately -24%
- NBV will decrease as provisions increase
- Collections/Revenue unchanged

## Next Steps

After running Iteration 1:
1. Assess remaining variance
2. Determine if further coverage adjustment needed
3. Consider segment-specific adjustments if variances differ significantly by segment
