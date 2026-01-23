# Calibration Results Summary

**Date:** 2026-01-22
**Iterations Completed:** 4

---

## Final Calibration Results (Iteration 4)

| Metric | Model | Budget | Variance | Status |
|--------|-------|--------|----------|--------|
| **Collections** | £142.8M | £140.2M | **+1.8%** | ✓ |
| **Revenue** | £50.3M | £51.5M | **-2.3%** | ✓ |
| **Closing GBV** | £174.3M | £182.0M | **-4.3%** | ✓ |
| **Closing NBV** | £121.3M | £142.9M | -15.2% | ✗ |
| **Net Impairment** | +£22.6M | -£28.8M | Sign mismatch | ✗ |

---

## Calibration Scaling Factors Applied

### Collections (Coll_Principal, Coll_Interest)

| Segment | Scale Factor | Business Logic |
|---------|--------------|----------------|
| NON PRIME | 0.927 | Reduce by 7% - model was collecting too aggressively |
| NRP-S | 0.912 | Reduce by 9% - align with budget collection rate |
| NRP-M | 0.86 | Reduce by 14% - significant over-collection in model |
| NRP-L | 0.86 | Same as NRP-M (combined budget segment) |
| PRIME | 0.66 | Reduce by 34% - model had ~2x budget collection rate |

### Revenue (InterestRevenue)

| Segment | Scale Factor | Business Logic |
|---------|--------------|----------------|
| NON PRIME | 1.01 | Slight increase to match budget yield |
| NRP-S | 0.97 | Slight reduction |
| NRP-M | 0.98 | Slight reduction |
| NRP-L | 0.98 | Same as NRP-M |
| PRIME | 0.92 | Reduce by 8% |

### Writeoffs (WO_DebtSold, WO_Other)

| Segment | Scale Factor | Business Logic |
|---------|--------------|----------------|
| ALL | 0.2 | Reduce by 80% to slow GBV runoff |

### Coverage Ratio (Total_Coverage_Ratio)

| Segment | Scale Factor | Business Logic |
|---------|--------------|----------------|
| ALL | 1.4 | Increase by 40% to approach budget coverage level |

---

## Impairment Methodology Difference - Key Finding

### The Sign Mismatch Explained

**Budget shows:** Negative impairment (£-28.8M) = Provision RELEASES = P&L benefit

**Model shows:** Positive impairment (+£22.6M) = Provision CHARGES = P&L cost

### Why This Happens

1. **Budget Assumption:** Coverage ratios DECREASE over time
   - Portfolio quality improves as loans season
   - Provisions are released as risk reduces
   - This generates P&L benefit (negative impairment)

2. **Model Assumption:** Coverage ratios STABLE or INCREASE with aging
   - Based on historical CohortAvg patterns
   - Older cohorts historically show higher coverage
   - This generates P&L cost (positive impairment)

### Business Implications

The budget likely uses a **forward-looking ECL approach** that assumes:
- Macroeconomic conditions will improve
- Collection performance will strengthen
- Portfolio quality will improve

The model uses a **historical pattern approach** that assumes:
- Future behavior mirrors past behavior
- Coverage ratios follow historical cohort curves
- No explicit macro assumptions

### Resolution Options

1. **Accept the difference** - Document that impairment methodology differs
2. **Manual overlay** - Apply a manual adjustment to impairment
3. **Change model approach** - Use declining coverage curves instead of historical

---

## Files Created

| File | Description |
|------|-------------|
| `Rate_Methodology_v7_CALIBRATED.csv` | Final calibrated methodology |
| `calibration_iter_1/` through `calibration_iter_4/` | Iteration outputs |
| `iterative_calibrate.py` | Calibration script |

---

## Recommended Next Steps

1. **For P&L forecasting:** Use calibrated Collections/Revenue (within 3% of budget)
2. **For Balance Sheet:** Use calibrated GBV (within 5% of budget)
3. **For Impairment:** Consider adding a manual overlay or adjusting the methodology to use forward-looking assumptions

---

## Calibration History

| Iteration | Collections Var | Revenue Var | GBV Var | Notes |
|-----------|-----------------|-------------|---------|-------|
| Original | +18.5% | +6.2% | -19.9% | Before calibration |
| 1 | -3.0% | +6.2% | -5.4% | Applied initial collection scales |
| 2 | -0.8% | -4.6% | -10.2% | Fine-tuned collection scales |
| 3 | +1.8% | -2.3% | -4.3% | Added writeoff reduction |
| 4 | +1.8% | -2.3% | -4.3% | Adjusted coverage ratio |

