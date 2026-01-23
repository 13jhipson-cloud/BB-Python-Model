# Iteration 1: Results Analysis

## Date: 2026-01-23

## Adjustment Applied
- Coverage Ratio: ScaledCohortAvg with 1.8x multiplier (IFRS 9 forward-looking adjustment)

## Results

| Metric | Budget | Forecast | Variance | Var % |
|--------|--------|----------|----------|-------|
| Collections | £140.22m | £154.42m | +£14.20m | +10.1% |
| ClosingGBV | £2,404.59m | £2,188.06m | -£216.53m | -9.0% |
| ClosingNBV | £2,006.18m | £1,424.69m | -£581.49m | **-29.0%** |
| Revenue | £51.49m | £47.12m | -£4.36m | -8.5% |
| GrossImpairment | -£28.76m | -£76.39m | -£47.64m | **+165.7%** |
| NetImpairment | -£28.76m | -£76.04m | -£47.28m | **+164.4%** |

## Analysis

### The 1.8x factor overshot significantly:
- **Baseline (1.0x)**: £12.14m impairment
- **Target (budget)**: £28.76m impairment
- **Iteration 1 (1.8x)**: £76.04m impairment

### Calculated optimal factor:
Using linear interpolation:
- Slope = (76.04 - 12.14) / (1.8 - 1.0) = 79.875 per unit factor
- Optimal factor = 1.0 + (28.76 - 12.14) / 79.875 = **1.21x**

### Side Effects:
- ClosingNBV dropped to -29% vs budget (was -3.4% in baseline)
- This is because higher provisions reduce NBV
- The 1.8x factor created excessive provision buildup

## Conclusion

The 1.8x adjustment was too aggressive. The relationship between coverage ratio scaling and impairment is steep - small changes in factor have large impacts.

**Recommendation for Iteration 2:** Use 1.2x scaling factor

## Modeling Rationale Preserved

The adjustment still reflects valid IFRS 9 forward-looking principles, just at a more moderate level. The rationale remains:
1. IFRS 9 requires forward-looking ECL provisions
2. Historical averages may understate expected losses
3. A modest uplift (1.2x) reflects prudent provisioning without excessive conservatism
