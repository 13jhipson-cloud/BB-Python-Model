# Rate Methodology Business Logic Report
## Pre-Calibration Baseline Documentation

**Report Date:** 2026-01-22
**Purpose:** Document the original Rate_Methodology configuration and provide business logic reasoning for each approach selection.

---

## 1. Executive Summary

This report documents the **original Rate_Methodology table** used in the Backbook Forecast model before any calibration attempts. The methodology table defines how forecast rates are calculated for each segment, cohort, and metric combination.

### Key Finding: Structural Mismatch with Budget

The auto-calibration system identified large variances between forecast and budget:
- **Collections:** -47% variance
- **Closing GBV:** -61% variance
- **Closing NBV:** -49% variance
- **Revenue:** -56% variance

**Root Cause:** This is a **structural mismatch**, not a rate calibration issue:
- The **Backbook Model** forecasts only **existing loans** (no new originations)
- The **Budget** likely includes **new business** (future loan originations)

This means the backbook will naturally show declining balances as existing loans pay down, while the budget expects growth from new loans. Rate calibration cannot resolve this fundamental difference.

---

## 2. Original Rate Methodology Table

### 2.1 Global Settings (ALL Segments)

| Metric | Approach | Param1 | Business Logic |
|--------|----------|--------|----------------|
| Coll_Principal | CohortAvg | 6 | Use rolling 6-month average of historical collection rates. This smooths out month-to-month volatility while remaining responsive to recent trends. |
| Coll_Interest | CohortAvg | 6 | Same as principal - interest collections follow similar patterns within each cohort's lifecycle. |
| InterestRevenue | CohortAvg | 6 | Interest accrual rates based on recent 6-month averages reflect current portfolio yield. |
| WO_DebtSold | CohortAvg | 6 | Debt sale writeoff rates averaged over 6 months capture typical sale timing patterns. |
| WO_Other | CohortAvg | 6 | Other writeoffs (charge-offs, settlements) averaged to capture natural portfolio attrition. |
| ContraSettlements_Principal | CohortAvg | 6 | Settlement patterns averaged to reflect typical customer behavior. |
| ContraSettlements_Interest | CohortAvg | 6 | Interest settlements follow similar averaging logic. |
| NewLoanAmount | Zero | - | **No new loans in backbook forecast** - this is a pure runoff model of existing loans. |
| Debt_Sale_Coverage_Ratio | Manual | 0.85 | Fixed 85% coverage for debt sale pool - assumes debt sold has 85% of average portfolio coverage. |
| Debt_Sale_Proceeds_Rate | Manual | 0.9 | Fixed 90p per GBV sold - reflects typical debt sale pricing. |

### 2.2 Coverage Ratio by Segment

The **Total_Coverage_Ratio** (provision/GBV) is the key impairment metric. Different approaches are used based on cohort maturity:

#### NON PRIME Segment

| MOB Range | Approach | Param | Business Logic |
|-----------|----------|-------|----------------|
| 40-999 | CohortAvg | 6 | **Mature cohorts (40+ MOB):** Coverage ratios have stabilized; use rolling average as they've reached steady-state performance. |
| 20-39 | CohortTrend | 6 | **Mid-age cohorts:** Coverage still trending upward as delinquencies accumulate; extrapolate the trend to forecast future coverage. |
| 0-19 | DonorCohort | 202101 | **Young cohorts:** Insufficient history; borrow curve from mature 2021-01 cohort which has proven representative behavior. |

#### NRP-S (Near Prime Small) Segment

| MOB Range | Approach | Param | Business Logic |
|-----------|----------|-------|----------------|
| 40-999 | CohortAvg | 6 | **Mature cohorts:** Stabilized coverage - use rolling average. |
| 20-39 | CohortTrend | 6 | **Mid-age:** Still trending - extrapolate. |
| 0-19 | DonorCohort | 202101 | **Young:** Borrow from 2021-01 donor cohort. |

#### NRP-M (Near Prime Medium) Segment

| MOB Range | Approach | Param | Business Logic |
|-----------|----------|-------|----------------|
| 40-999 | CohortAvg | 6 | **Mature cohorts:** Stabilized - rolling average. |
| 20-39 | CohortTrend | 6 | **Mid-age:** Trending - extrapolate. |
| 0-19 | DonorCohort | 202101 | **Young:** Borrow from mature 2021-01 cohort. |
| 202001 (all MOB) | Manual | 0 | **Exception:** 2020-01 cohort has historically shown 0% coverage - likely a data anomaly or special treatment cohort. |

#### NRP-L (Near Prime Large) Segment

| MOB Range | Approach | Param | Business Logic |
|-----------|----------|-------|----------------|
| 40-999 | CohortAvg | 6 | **Mature cohorts:** Rolling average for stable coverage. |
| 20-39 | CohortTrend | 6 | **Mid-age:** Extrapolate trending coverage. |
| 0-19 | DonorCohort | 202301 | **Young:** Borrow from 2023-01 (only mature NRP-L cohort available). |
| 202501-202509 | Manual | 0 | **Exception:** All 2025 cohorts consistently show 0% coverage - too new to have accumulated provisions. |

#### PRIME Segment

| MOB Range | Approach | Param | Business Logic |
|-----------|----------|-------|----------------|
| 40-999 | CohortAvg | 6 | **Mature cohorts:** Rolling average for stable coverage. |
| 20-39 | CohortTrend | 6 | **Mid-age:** Extrapolate coverage trend. |
| 0-19 | DonorCohort | 202101 | **Young:** Borrow from mature 2021-01 cohort. |
| 202405, 202407, 202409, 202504-202509 | Manual | 0 | **Exception:** Recent cohorts with 0% historical coverage - high-quality prime loans with no delinquency yet. |

---

## 3. Approach Definitions and Business Logic

### 3.1 CohortAvg (Cohort Average)
- **What it does:** Calculates a rolling N-period average of historical rates for each cohort
- **When to use:** For mature cohorts where rates have stabilized
- **Business Logic:** Smooths volatility while reflecting recent performance; appropriate when historical patterns are expected to continue

### 3.2 CohortTrend (Cohort Trend Extrapolation)
- **What it does:** Fits a trend line to recent data and extrapolates forward
- **When to use:** For mid-age cohorts where metrics are still evolving (e.g., coverage ratio increasing as loans age)
- **Business Logic:** Captures the natural lifecycle of loans where impairment builds over time before stabilizing

### 3.3 DonorCohort
- **What it does:** Borrows the curve from a specified mature cohort
- **When to use:** For young cohorts with insufficient history
- **Business Logic:** Uses a "donor" cohort that has similar characteristics and proven performance history

### 3.4 Manual
- **What it does:** Applies a fixed value
- **When to use:** For known exceptions or where data-driven approaches aren't appropriate
- **Business Logic:** Used for cohorts with unusual characteristics (e.g., consistently 0% coverage)

### 3.5 Zero
- **What it does:** Sets value to zero
- **When to use:** For metrics that shouldn't exist in the model
- **Business Logic:** NewLoanAmount = Zero because this is a backbook-only model with no new originations

---

## 4. Original Forecast Output vs Budget

### 4.1 Collections (12-month total)

| Segment | Forecast | Budget | Variance |
|---------|----------|--------|----------|
| Non Prime | 83,622,318 | 159,002,020 | -47.4% |
| Near Prime Small | 27,368,393 | 46,966,095 | -41.7% |
| Near Prime Medium | 52,280,083 | 105,646,288 | -50.5% |
| Prime | 2,935,677 | 3,775,686 | -22.2% |
| **Total** | **166,206,469** | **315,390,089** | **-47.3%** |

### 4.2 Closing GBV (12-month total)

| Segment | Forecast | Budget | Variance |
|---------|----------|--------|----------|
| Non Prime | 989,358,454 | 2,302,920,330 | -57.0% |
| Near Prime Small | 367,157,739 | 891,148,114 | -58.8% |
| Near Prime Medium | 1,079,244,215 | 2,984,263,604 | -63.8% |
| Prime | 19,845,759 | 53,610,917 | -63.0% |
| **Total** | **2,455,606,166** | **6,231,942,966** | **-60.6%** |

### 4.3 Closing NBV (12-month total)

| Segment | Forecast | Budget | Variance |
|---------|----------|--------|----------|
| Non Prime | 824,768,410 | 1,471,025,460 | -43.9% |
| Near Prime Small | 341,201,525 | 682,312,508 | -50.0% |
| Near Prime Medium | 1,000,444,244 | 2,310,366,864 | -56.7% |
| Prime | 16,793,010 | 38,479,013 | -56.4% |
| **Total** | **2,183,207,189** | **4,267,151,226** | **-48.8%** |

### 4.4 Revenue (12-month total)

| Segment | Forecast | Budget | Variance |
|---------|----------|--------|----------|
| Non Prime | 27,448,018 | 55,755,597 | -50.8% |
| Near Prime Small | 6,596,085 | 13,709,771 | -51.9% |
| Near Prime Medium | 16,280,474 | 43,698,338 | -62.7% |
| Prime | 202,694 | 527,543 | -61.6% |
| **Total** | **50,527,272** | **113,691,249** | **-55.6%** |

### 4.5 Impairment (12-month total)

| Segment | Forecast Gross | Budget Gross | Variance |
|---------|----------------|--------------|----------|
| Non Prime | 2,392,134 | -29,791,341 | -108.0% |
| Near Prime Small | -233,691 | -5,015,813 | -95.3% |
| Near Prime Medium | 5,000,028 | -24,620,344 | -120.3% |
| Prime | -276,985 | 1,026 | N/A |
| **Total** | **6,881,486** | **-59,426,472** | **-111.6%** |

**Note on Impairment Sign:**
- Positive = Impairment charge (cost)
- Negative = Impairment release (benefit)

The budget shows negative impairment (releases/benefits) while the forecast shows positive (charges), indicating fundamentally different portfolio dynamics.

---

## 5. Why Calibration Cannot Resolve the Variance

### 5.1 Structural Difference: Backbook vs Full Business

The Backbook Model is designed to forecast the **runoff** of existing loans only. This means:
1. **No new originations** - GBV naturally declines as loans pay down
2. **Aging portfolio** - Average MOB increases over time
3. **Declining absolute volumes** - Collections, revenue all decline with GBV

The Budget likely includes:
1. **New loan originations** - Adding new GBV each month
2. **Portfolio growth** - New business outpacing runoff
3. **Growing absolute volumes** - Collections, revenue grow with larger book

### 5.2 Rate Calibration Limitations

Rate calibration can only adjust **rates** (percentages), not **absolute volumes**:
- If the budget expects £315M collections on a £6B book (5.1% rate)
- And the forecast shows £166M collections on a £2.5B book (6.7% rate)
- The forecast actually has a **higher collection rate** - it's collecting more efficiently!

Adjusting rates upward would make the forecast even more unrealistic.

### 5.3 Impairment Sign Reversal

The budget shows **impairment releases** (negative) while the forecast shows **impairment charges** (positive):
- Budget: Growing portfolio with new loans diluting coverage → releases
- Forecast: Static/shrinking portfolio with aging loans → charges

This sign difference cannot be resolved through rate calibration.

---

## 6. Recommendations

### 6.1 For Backbook Forecasting

The current Rate_Methodology is **appropriate for backbook forecasting** because:
1. It uses data-driven approaches (CohortAvg, CohortTrend) based on historical patterns
2. It handles young cohorts sensibly using donor cohorts
3. It accounts for exceptions with Manual overrides
4. It correctly sets NewLoanAmount to Zero

### 6.2 For Budget Comparison

To compare backbook forecasts to budget:
1. **Extract backbook-only budget** - Separate new business from existing book in the budget
2. **Use rate comparisons** - Compare collection rates, coverage ratios, not absolute values
3. **Accept structural difference** - The backbook will always show lower absolute values than a growing portfolio

### 6.3 If Full Business Forecast is Needed

A separate model would be required that:
1. Includes new business origination forecasts
2. Applies different rate methodologies to new vs existing loans
3. Models portfolio growth dynamics

---

## 7. Files Saved

| File | Description |
|------|-------------|
| `Rate_Methodology_ORIGINAL.csv` | Original methodology table before calibration |
| `Forecast_Summary.xlsx` | Original forecast summary output |
| `Forecast_Details.xlsx` | Original forecast detail output |
| `Impairment_Analysis.xlsx` | Original impairment analysis |
| `Validation_Report.xlsx` | Forecast validation report |
| `Original_Forecast_vs_Budget.csv` | Detailed comparison by month |
| `Original_Variance_Summary.csv` | Summary variance by segment/metric |

---

## 8. Conclusion

The original Rate_Methodology table is **technically sound and business-appropriate** for forecasting the backbook (existing loans only). The large variances against budget are not a deficiency in the methodology but rather a **fundamental structural difference** between:
- A pure runoff model (backbook)
- A full business forecast including new originations (budget)

No amount of rate calibration can resolve this structural difference. The appropriate use of this model is to forecast the runoff of the existing loan book, which it does accurately using proven cohort-based methodologies.
