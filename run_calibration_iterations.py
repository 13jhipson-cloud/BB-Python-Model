#!/usr/bin/env python3
"""
Run Calibration Iterations with Documented Adjustments

This script runs multiple iterations of the backbook forecast model,
starting from the original methodology (with modeling logic) and making
justified adjustments based on variance analysis.

Each iteration is fully documented with:
- The methodology used
- The variance analysis vs budget
- The modeling rationale for any adjustments
"""

import pandas as pd
import numpy as np
import subprocess
import os
import sys
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = '/home/user/BB-Python-Model'
FACT_RAW = os.path.join(BASE_DIR, 'Fact_Raw_Transformed.csv')  # Use actual portfolio data
BUDGET_FILE = os.path.join(BASE_DIR, 'Budget consol file.xlsx')
ORIGINAL_METHODOLOGY = os.path.join(BASE_DIR, 'pre_calibration_baseline', 'Rate_Methodology_ORIGINAL.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'calibration_iterations')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Segment mapping
SEGMENT_MAPPING = {
    'NON PRIME': 'Non Prime',
    'NRP-S': 'Near Prime Small',
    'NRP-M': 'Near Prime Medium',
    'NRP-L': 'Near Prime Medium',  # Combined with NRP-M in budget
    'PRIME': 'Prime',
}

# Budget row indices (from compare_to_budget.py)
BUDGET_ROWS = {
    'Collections': {'Non Prime': 11, 'Near Prime Small': 12, 'Near Prime Medium': 13, 'Prime': 14, 'Total': 15},
    'ClosingGBV': {'Non Prime': 22, 'Near Prime Small': 23, 'Near Prime Medium': 24, 'Prime': 25, 'Total': 26},
    'ClosingNBV': {'Non Prime': 42, 'Near Prime Small': 43, 'Near Prime Medium': 44, 'Prime': 45, 'Total': 46},
    'Revenue': {'Non Prime': 62, 'Near Prime Small': 63, 'Near Prime Medium': 64, 'Prime': 65, 'Total': 66},
    'GrossImpairment': {'Non Prime': 73, 'Near Prime Small': 74, 'Near Prime Medium': 75, 'Prime': 76, 'Total': 77},
    'DebtSaleGain': {'Non Prime': 105, 'Near Prime Small': 106, 'Near Prime Medium': 107, 'Prime': 108, 'Total': 109},
    'NetImpairment': {'Non Prime': 121, 'Near Prime Small': 122, 'Near Prime Medium': 123, 'Prime': 124, 'Total': 125},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_budget():
    """Load budget data from Excel file."""
    print("Loading budget data...")
    xl = pd.ExcelFile(BUDGET_FILE)
    df_raw = pd.read_excel(xl, sheet_name='P&L analysis - BB', header=None)

    # Extract dates from row 2, columns 3 onwards
    dates = pd.to_datetime(df_raw.iloc[2, 3:].values)

    # Build long-format dataframe
    records = []
    for metric, segment_rows in BUDGET_ROWS.items():
        for segment, row_idx in segment_rows.items():
            values = df_raw.iloc[row_idx, 3:].values
            for date, value in zip(dates, values):
                if pd.notna(value):
                    records.append({
                        'Date': date,
                        'Segment': segment,
                        'Metric': metric,
                        'Budget_Value': float(value)
                    })

    budget_df = pd.DataFrame(records)
    print(f"  Loaded {len(budget_df)} budget data points")
    print(f"  Date range: {budget_df['Date'].min()} to {budget_df['Date'].max()}")
    return budget_df


def load_forecast_output():
    """Load forecast output from the model run."""
    # Load summary file
    summary_path = os.path.join(MODEL_OUTPUT_DIR, 'Forecast_Summary.xlsx')
    impairment_path = os.path.join(MODEL_OUTPUT_DIR, 'Impairment_Analysis.xlsx')

    if not os.path.exists(summary_path):
        print(f"ERROR: Forecast summary not found at {summary_path}")
        return None

    df_summary = pd.read_excel(summary_path, sheet_name='Summary')
    df_summary['ForecastMonth'] = pd.to_datetime(df_summary['ForecastMonth'])

    # Load impairment details
    if os.path.exists(impairment_path):
        df_impairment = pd.read_excel(impairment_path, sheet_name='Impairment_Detail')
        df_impairment['ForecastMonth'] = pd.to_datetime(df_impairment['ForecastMonth'])
    else:
        df_impairment = None

    # Map segments
    df_summary['BudgetSegment'] = df_summary['Segment'].map(SEGMENT_MAPPING)

    # Aggregate by budget segment and month
    agg_df = df_summary.groupby(['ForecastMonth', 'BudgetSegment']).agg({
        'OpeningGBV': 'sum',
        'ClosingGBV': 'sum',
        'ClosingNBV': 'sum',
        'InterestRevenue': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'Net_Impairment': 'sum',
    }).reset_index()

    # Add impairment data if available
    if df_impairment is not None:
        df_impairment['BudgetSegment'] = df_impairment['Segment'].map(SEGMENT_MAPPING)
        imp_agg = df_impairment.groupby(['ForecastMonth', 'BudgetSegment']).agg({
            'Gross_Impairment_ExcludingDS': 'sum',
            'Debt_Sale_Impact': 'sum',
        }).reset_index()
        agg_df = agg_df.merge(imp_agg, on=['ForecastMonth', 'BudgetSegment'], how='left')

    # Calculate derived metrics
    agg_df['Collections'] = -(agg_df['Coll_Principal'] + agg_df['Coll_Interest'])
    agg_df['Revenue'] = agg_df['InterestRevenue']

    if 'Gross_Impairment_ExcludingDS' in agg_df.columns:
        agg_df['GrossImpairment'] = agg_df['Gross_Impairment_ExcludingDS']
        agg_df['DebtSaleGain'] = agg_df['Debt_Sale_Impact']
    else:
        agg_df['GrossImpairment'] = agg_df['Net_Impairment']  # Fallback
        agg_df['DebtSaleGain'] = 0

    agg_df['NetImpairment'] = agg_df['Net_Impairment']

    # Reshape to long format
    records = []
    metrics = ['Collections', 'ClosingGBV', 'ClosingNBV', 'Revenue', 'GrossImpairment', 'DebtSaleGain', 'NetImpairment']

    for _, row in agg_df.iterrows():
        for metric in metrics:
            records.append({
                'Date': row['ForecastMonth'],
                'Segment': row['BudgetSegment'],
                'Metric': metric,
                'Forecast_Value': row[metric]
            })

    # Calculate totals
    total_df = agg_df.groupby('ForecastMonth')[metrics].sum().reset_index()
    for _, row in total_df.iterrows():
        for metric in metrics:
            records.append({
                'Date': row['ForecastMonth'],
                'Segment': 'Total',
                'Metric': metric,
                'Forecast_Value': row[metric]
            })

    return pd.DataFrame(records)


def compare_to_budget(forecast_df, budget_df):
    """Compare forecast to budget and calculate variances."""
    # Find common dates
    budget_dates = set(budget_df['Date'].unique())
    forecast_dates = set(forecast_df['Date'].unique())
    common_dates = budget_dates & forecast_dates

    if not common_dates:
        print("ERROR: No common dates between budget and forecast")
        return None

    print(f"  Comparing {len(common_dates)} common months")

    budget_filtered = budget_df[budget_df['Date'].isin(common_dates)]
    forecast_filtered = forecast_df[forecast_df['Date'].isin(common_dates)]

    # Merge
    merged = pd.merge(budget_filtered, forecast_filtered,
                      on=['Date', 'Segment', 'Metric'], how='outer')

    # Calculate variances
    merged['Variance'] = merged['Forecast_Value'] - merged['Budget_Value']
    merged['Variance_Pct'] = np.where(
        merged['Budget_Value'] != 0,
        (merged['Forecast_Value'] / merged['Budget_Value'] - 1) * 100,
        np.nan
    )
    merged['Abs_Variance'] = merged['Variance'].abs()

    return merged.sort_values(['Date', 'Segment', 'Metric'])


# =============================================================================
# MODEL EXECUTION
# =============================================================================

def run_model(methodology_file):
    """Run the backbook forecast model."""
    cmd = [
        'python', os.path.join(BASE_DIR, 'backbook_forecast.py'),
        '--fact-raw', FACT_RAW,
        '--methodology', methodology_file,
        '--months', '12'
    ]

    print(f"\nRunning model with methodology: {os.path.basename(methodology_file)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"ERROR: Model failed")
        print(result.stderr[-1500:])
        return False

    # Print last part of output
    output_lines = result.stdout.strip().split('\n')
    print("  Model completed. Last 10 lines of output:")
    for line in output_lines[-10:]:
        print(f"    {line}")

    return True


# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def analyze_historical_rates():
    """Analyze historical rate patterns to inform methodology decisions."""
    df = pd.read_csv(FACT_RAW)
    df['Date'] = pd.to_datetime(df['CalendarMonth'])

    # Get last 12 months
    max_date = df['Date'].max()
    recent = df[df['Date'] >= max_date - pd.DateOffset(months=12)].copy()

    analysis = {}

    for segment in df['Segment'].unique():
        seg_df = recent[recent['Segment'] == segment].copy()
        if len(seg_df) == 0:
            continue

        seg_analysis = {}

        # Collection rate trends
        if 'Coll_Principal' in seg_df.columns and 'OpeningGBV' in seg_df.columns:
            seg_df['coll_rate'] = seg_df['Coll_Principal'].abs() / seg_df['OpeningGBV'].replace(0, np.nan)
            monthly_rates = seg_df.groupby('Date')['coll_rate'].mean()
            seg_analysis['avg_coll_rate'] = monthly_rates.mean()
            seg_analysis['coll_rate_trend'] = monthly_rates.diff().mean()  # Positive = increasing
            seg_analysis['coll_rate_last_3m'] = monthly_rates.tail(3).mean()
            seg_analysis['coll_rate_first_3m'] = monthly_rates.head(3).mean()

        # Interest revenue rate
        if 'InterestRevenue' in seg_df.columns and 'OpeningGBV' in seg_df.columns:
            seg_df['int_rate'] = seg_df['InterestRevenue'] / seg_df['OpeningGBV'].replace(0, np.nan)
            monthly_int = seg_df.groupby('Date')['int_rate'].mean()
            seg_analysis['avg_int_rate'] = monthly_int.mean()

        # Coverage ratio trends
        if 'Total_Coverage_Ratio' in seg_df.columns:
            monthly_cr = seg_df.groupby('Date')['Total_Coverage_Ratio'].mean()
            seg_analysis['avg_coverage'] = monthly_cr.mean()
            seg_analysis['coverage_trend'] = monthly_cr.diff().mean()
            seg_analysis['coverage_last'] = monthly_cr.iloc[-1] if len(monthly_cr) > 0 else np.nan

        analysis[segment] = seg_analysis

    return analysis


def generate_variance_summary(comparison_df):
    """Generate summary of variances by metric and segment."""
    summary = comparison_df.groupby(['Metric', 'Segment']).agg({
        'Budget_Value': 'sum',
        'Forecast_Value': 'sum',
        'Variance': 'sum',
        'Abs_Variance': 'sum'
    }).reset_index()

    summary['Total_Variance_Pct'] = np.where(
        summary['Budget_Value'] != 0,
        (summary['Forecast_Value'] / summary['Budget_Value'] - 1) * 100,
        np.nan
    )

    return summary


def print_variance_summary(summary_df):
    """Print variance summary to console."""
    print("\n" + "=" * 100)
    print("VARIANCE SUMMARY BY METRIC AND SEGMENT")
    print("=" * 100)

    for metric in ['Collections', 'ClosingGBV', 'ClosingNBV', 'Revenue', 'GrossImpairment', 'NetImpairment']:
        metric_df = summary_df[summary_df['Metric'] == metric].copy()
        if len(metric_df) == 0:
            continue

        print(f"\n{metric}:")
        print(f"  {'Segment':<20} {'Budget':>15} {'Forecast':>15} {'Variance':>15} {'Var %':>10}")
        print("  " + "-" * 78)

        for _, row in metric_df.iterrows():
            budget = row['Budget_Value'] / 1e6
            forecast = row['Forecast_Value'] / 1e6
            variance = row['Variance'] / 1e6
            var_pct = row['Total_Variance_Pct']

            flag = "⚠️ " if abs(var_pct) > 10 else "   "
            print(f"  {row['Segment']:<20} £{budget:>12.2f}m £{forecast:>12.2f}m £{variance:>+12.2f}m {var_pct:>+8.1f}% {flag}")


def save_iteration(iteration_num, methodology_df, comparison_df, summary_df, report_text, adjustments_text):
    """Save all artifacts for this iteration."""
    iter_dir = os.path.join(OUTPUT_DIR, f'iteration_{iteration_num}')
    os.makedirs(iter_dir, exist_ok=True)

    # Save methodology
    methodology_df.to_csv(os.path.join(iter_dir, 'Rate_Methodology.csv'), index=False)

    # Save comparison detail
    if comparison_df is not None:
        comparison_df.to_csv(os.path.join(iter_dir, 'Variance_Analysis_Detail.csv'), index=False)

    # Save summary
    if summary_df is not None:
        summary_df.to_csv(os.path.join(iter_dir, 'Variance_Summary.csv'), index=False)

    # Copy forecast outputs
    for fname in ['Forecast_Summary.xlsx', 'Impairment_Analysis.xlsx', 'Rate_Analysis.xlsx']:
        src = os.path.join(MODEL_OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, iter_dir)

    # Save report
    with open(os.path.join(iter_dir, 'ITERATION_REPORT.md'), 'w') as f:
        f.write(report_text)

    # Save adjustments documentation
    with open(os.path.join(iter_dir, 'ADJUSTMENTS.md'), 'w') as f:
        f.write(adjustments_text)

    print(f"\n✓ Iteration {iteration_num} saved to: {iter_dir}")
    return iter_dir


# =============================================================================
# ITERATION 0: BASELINE (ORIGINAL METHODOLOGY)
# =============================================================================

def run_iteration_0():
    """Run baseline iteration with original methodology (no adjustments)."""
    print("\n" + "=" * 100)
    print("ITERATION 0: BASELINE (Original Methodology with Corrected Signage)")
    print("=" * 100)

    # Load original methodology
    methodology_df = pd.read_csv(ORIGINAL_METHODOLOGY)
    print(f"\nLoaded original methodology: {len(methodology_df)} rules")

    # Save methodology to working location
    working_methodology = os.path.join(OUTPUT_DIR, 'iteration_0', 'Rate_Methodology.csv')
    os.makedirs(os.path.dirname(working_methodology), exist_ok=True)
    methodology_df.to_csv(working_methodology, index=False)

    # Also copy to sample_data for the model to use
    model_methodology = os.path.join(BASE_DIR, 'sample_data', 'Rate_Methodology.csv')
    methodology_df.to_csv(model_methodology, index=False)

    # Run model
    if not run_model(model_methodology):
        return None, None, None

    # Load budget and forecast
    budget_df = load_budget()
    forecast_df = load_forecast_output()

    if forecast_df is None:
        return None, None, None

    # Compare
    comparison_df = compare_to_budget(forecast_df, budget_df)
    summary_df = generate_variance_summary(comparison_df)

    # Print summary
    print_variance_summary(summary_df)

    # Generate report
    report = f"""# Iteration 0: Baseline Analysis

## Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

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
"""

    # Add observations based on summary
    for metric in ['Collections', 'Revenue', 'GrossImpairment', 'ClosingGBV']:
        metric_total = summary_df[(summary_df['Metric'] == metric) & (summary_df['Segment'] == 'Total')]
        if len(metric_total) > 0:
            var_pct = metric_total['Total_Variance_Pct'].values[0]
            report += f"\n- **{metric}**: Forecast is {var_pct:+.1f}% vs budget"

    report += """

## Next Steps
Analyze the variances to determine appropriate modeling-based adjustments.
"""

    adjustments = """# Iteration 0: No Adjustments

This is the baseline iteration. No adjustments have been made.

The original methodology is based on pure modeling logic:
- Historical rate averages for stable metrics
- Trend extrapolation for seasoning metrics
- Donor cohort approaches for young cohorts

The variances from this baseline will inform what adjustments (if any) are needed,
and importantly, what the **modeling rationale** for those adjustments should be.
"""

    # Save iteration
    save_iteration(0, methodology_df, comparison_df, summary_df, report, adjustments)

    return methodology_df, comparison_df, summary_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 100)
    print("CALIBRATION ITERATION RUNNER")
    print("Starting with original methodology and applying justified adjustments")
    print("=" * 100)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Analyze historical rates first
    print("\n" + "-" * 50)
    print("Analyzing historical rate patterns...")
    print("-" * 50)
    hist_analysis = analyze_historical_rates()

    print("\nHistorical Rate Analysis (last 12 months):")
    for segment, metrics in hist_analysis.items():
        print(f"\n  {segment}:")
        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                print(f"    {key}: {value:.4f}")

    # Run Iteration 0 (baseline)
    methodology_df, comparison_df, summary_df = run_iteration_0()

    if methodology_df is None:
        print("\nERROR: Baseline iteration failed")
        return

    print("\n" + "=" * 100)
    print("ITERATION 0 COMPLETE")
    print("=" * 100)
    print("\nReview the variance analysis to determine justified adjustments for Iteration 1")
    print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'iteration_0')}")


if __name__ == '__main__':
    main()
