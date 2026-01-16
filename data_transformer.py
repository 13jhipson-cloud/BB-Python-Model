#!/usr/bin/env python3
"""
Data Transformer Module

Transforms raw Fact_Raw_New.xlsx data into the format expected by the backbook forecast model.
Replicates the Power Query M code transformations from Fact_Raw_Mcode.txt.

Key transformations:
1. Column renaming to match model expectations
2. Calendar month to end-of-month date conversion
3. Cohort clustering (Backbook 1-4 groupings)
4. Segment creation (NRP-S, NRP-M, NRP-L from Near Prime + LoanSize)
5. MOB calculation from original cohort date
6. Grouping by CalendarMonth, Cohort, Segment, MOB

Author: Claude Code
Version: 1.0.0
"""

import logging
import re
from calendar import monthrange
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Column renaming map (raw → model format)
COLUMN_RENAME_MAP = {
    'cohort': 'Cohort_Raw',
    'calendarmonth': 'CalendarMonth_Raw',
    'lob': 'LOB',
    'loansize': 'LoanSize',
    'openinggbv': 'OpeningGBV',
    'disbursalsexcltopup': 'Disb_ExclTopups',
    'disbursalstopup': 'TopUp_IncrCash',
    'loanamount': 'NewLoanAmount',
    'principalcollections': 'Coll_Principal',
    'interestcollections': 'Coll_Interest',
    'principalcontrasettlement': 'ContraSettlements_Principal',
    'nonprincipalcontrasettlement': 'ContraSettlements_Interest',
    'debtsalewriteoffs': 'WO_DebtSold',
    'otherwriteoffs': 'WO_Other',
    'closinggbv': 'ClosingGBV_Reported',
    'interestrevenue': 'InterestRevenue',
    'provisionatmonthend': 'Provision_Balance',
    'debtsaleproceeds': 'Debt_Sale_Proceeds',
}

# Numeric columns for aggregation
NUMERIC_COLUMNS = [
    'OpeningGBV',
    'Disb_ExclTopups',
    'TopUp_IncrCash',
    'NewLoanAmount',
    'Coll_Principal',
    'Coll_Interest',
    'ContraSettlements_Principal',
    'ContraSettlements_Interest',
    'WO_DebtSold',
    'WO_Other',
    'ClosingGBV_Reported',
    'InterestRevenue',
    'Provision_Balance',
    'Debt_Sale_Proceeds',
]


def yyyymm_to_end_of_month(yyyymm: int) -> date:
    """
    Convert YYYYMM integer to end-of-month date.

    Args:
        yyyymm: Integer in YYYYMM format (e.g., 202401)

    Returns:
        date: End of month date
    """
    year = yyyymm // 100
    month = yyyymm % 100
    _, last_day = monthrange(year, month)
    return date(year, month, last_day)


def parse_cohort_ym(cohort_val) -> int:
    """
    Parse cohort value to YYYYMM integer.

    Handles:
    - Integer YYYYMM format
    - String YYYYMM format
    - "PRE-2020" or "PRE 2020" → -1 (special marker for 201912)

    Args:
        cohort_val: Raw cohort value

    Returns:
        int: YYYYMM integer or -1 for pre-2020
    """
    if pd.isna(cohort_val):
        return None

    # Convert to string and clean
    cohort_str = str(cohort_val).strip().upper()

    # Check for PRE-2020 pattern
    if 'PRE' in cohort_str and '2020' in cohort_str:
        return -1

    # Try to parse as integer
    try:
        return int(float(cohort_val))
    except (ValueError, TypeError):
        pass

    # Try to parse as date string
    try:
        dt = pd.to_datetime(cohort_val)
        return dt.year * 100 + dt.month
    except Exception:
        pass

    logger.warning(f"Could not parse cohort value: {cohort_val}")
    return None


def get_cohort_cluster(cohort_ym: int) -> int:
    """
    Map cohort YYYYMM to clustered cohort based on Backbook groupings.

    Clustering rules:
    - PRE-2020 (-1) → 201912
    - 202001-202012 → 202001 (Backbook 4)
    - 202101-202208 → 202101 (Backbook 3)
    - 202209-202305 → 202201 (Backbook 2)
    - 202306-202403 → 202301 (Backbook 1)
    - Others → keep original (monthly cohorts from 202404+)

    Args:
        cohort_ym: Original YYYYMM value

    Returns:
        int: Clustered cohort YYYYMM
    """
    if cohort_ym is None:
        return None

    if cohort_ym == -1:
        return 201912  # PRE-2020

    if 202001 <= cohort_ym <= 202012:
        return 202001  # Backbook 4

    if 202101 <= cohort_ym <= 202208:
        return 202101  # Backbook 3

    if 202209 <= cohort_ym <= 202305:
        return 202201  # Backbook 2

    if 202306 <= cohort_ym <= 202403:
        return 202301  # Backbook 1

    return cohort_ym  # Monthly cohorts (202404+)


def parse_loan_size_bucket(loan_size: str) -> str:
    """
    Parse loan size string to S/M/L bucket.

    Rules (from M code):
    - £0-5k → S (Small)
    - £5-10k or £10-15k → M (Medium)
    - £15-20k → L (Large)

    Args:
        loan_size: Raw loan size string (e.g., "£0-£5k", "£5k-£10k")

    Returns:
        str: 'S', 'M', 'L', or '' if cannot parse
    """
    if pd.isna(loan_size):
        return ''

    # Extract digits and hyphens
    raw = re.sub(r'[^0-9\-]', '', str(loan_size))
    parts = raw.split('-')

    if len(parts) < 2:
        return ''

    try:
        low = int(parts[0])
        high = int(parts[1])
    except (ValueError, IndexError):
        return ''

    if low < 5:
        return 'S'  # 0-5k → NRP-S
    elif low >= 5 and high <= 15:
        return 'M'  # 5-10k or 10-15k → NRP-M
    elif low >= 15:
        return 'L'  # 15-20k → NRP-L
    else:
        return ''


def build_segment(lob: str, loan_size: str) -> str:
    """
    Build segment from LOB and LoanSize.

    Rules:
    - Near Prime + size bucket → NRP-S, NRP-M, or NRP-L
    - Non Prime → NON PRIME
    - Prime → PRIME

    Args:
        lob: Line of Business
        loan_size: Loan size string

    Returns:
        str: Segment name
    """
    if pd.isna(lob):
        return ''

    lob_clean = str(lob).strip().upper().replace('-', ' ')

    if lob_clean == 'NEAR PRIME':
        size_bucket = parse_loan_size_bucket(loan_size)
        if size_bucket:
            return f'NRP-{size_bucket}'
        return 'NEAR PRIME'  # Fallback if no size bucket

    return lob_clean


def calculate_mob(calendar_month_raw: int, cohort_date: date) -> int:
    """
    Calculate Months on Book from cohort date to calendar month.

    Args:
        calendar_month_raw: Calendar month as YYYYMM integer
        cohort_date: Original cohort date (not clustered)

    Returns:
        int: Months on Book
    """
    cal_year = calendar_month_raw // 100
    cal_month = calendar_month_raw % 100

    mob = (cal_year * 12 + cal_month) - (cohort_date.year * 12 + cohort_date.month)
    return mob


def transform_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Fact_Raw_New data into model-ready format.

    This replicates the Power Query M code transformations:
    1. Rename columns
    2. Convert calendar month to end-of-month date
    3. Parse and cluster cohorts
    4. Build segments from LOB + LoanSize
    5. Calculate MOB from original cohort date
    6. Group and aggregate by CalendarMonth, Cohort, Segment, MOB
    7. Add DaysInMonth
    8. Replace nulls with zeros

    Args:
        df_raw: Raw dataframe from Fact_Raw_New.xlsx

    Returns:
        pd.DataFrame: Transformed dataframe ready for model
    """
    logger.info(f"Starting transformation of {len(df_raw)} raw rows...")

    # Step 1: Rename columns
    df = df_raw.rename(columns=COLUMN_RENAME_MAP).copy()
    logger.info("Renamed columns to model format")

    # Step 2: Parse cohort YYYYMM from raw cohort
    df['CohortYM'] = df['Cohort_Raw'].apply(parse_cohort_ym)

    # Step 3: Create CohortDate (original cohort date, not clustered)
    def cohort_ym_to_date(ym):
        if ym is None:
            return None
        if ym == -1:
            return date(2019, 12, 31)  # PRE-2020
        year = ym // 100
        month = ym % 100
        return date(year, month, 1)

    df['CohortDate'] = df['CohortYM'].apply(cohort_ym_to_date)

    # Step 4: Apply cohort clustering
    df['CohortCluster'] = df['CohortYM'].apply(get_cohort_cluster)
    df['Cohort'] = df['CohortCluster'].astype(str)
    logger.info("Applied cohort clustering (Backbook 1-4)")

    # Step 5: Build Segment from LOB + LoanSize
    df['Segment'] = df.apply(
        lambda row: build_segment(row.get('LOB'), row.get('LoanSize')),
        axis=1
    )
    logger.info("Built segments from LOB + LoanSize")

    # Step 6: Calculate MOB from original CohortDate
    df['MOB'] = df.apply(
        lambda row: calculate_mob(row['CalendarMonth_Raw'], row['CohortDate'])
        if row['CohortDate'] is not None else None,
        axis=1
    )

    # Filter out negative MOB
    df = df[df['MOB'] >= 0].copy()
    logger.info(f"Calculated MOB, {len(df)} rows with MOB >= 0")

    # Step 7: Convert CalendarMonth to end-of-month date
    df['CalendarMonth'] = df['CalendarMonth_Raw'].apply(
        lambda x: pd.Timestamp(yyyymm_to_end_of_month(x))
    )

    # Step 8: Add DaysInMonth
    df['DaysInMonth'] = df['CalendarMonth'].dt.days_in_month

    # Step 9: Fill missing numeric values with 0
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Step 10: Group by CalendarMonth, Cohort, Segment, MOB and aggregate
    group_cols = ['CalendarMonth', 'Cohort', 'Segment', 'MOB']

    # Build aggregation dict
    agg_dict = {col: 'sum' for col in NUMERIC_COLUMNS if col in df.columns}
    agg_dict['DaysInMonth'] = 'mean'

    df_grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Round DaysInMonth to int
    df_grouped['DaysInMonth'] = df_grouped['DaysInMonth'].round().astype(int)

    logger.info(f"Grouped to {len(df_grouped)} rows by Cohort × Segment × CalendarMonth × MOB")

    # Step 11: Sort for consistency
    df_grouped = df_grouped.sort_values(
        ['Segment', 'Cohort', 'CalendarMonth', 'MOB']
    ).reset_index(drop=True)

    # Print summary statistics
    logger.info(f"Unique Segments: {df_grouped['Segment'].unique().tolist()}")
    logger.info(f"Unique Cohorts: {sorted(df_grouped['Cohort'].unique().tolist())}")
    logger.info(f"MOB range: {df_grouped['MOB'].min()} to {df_grouped['MOB'].max()}")
    logger.info(f"Date range: {df_grouped['CalendarMonth'].min()} to {df_grouped['CalendarMonth'].max()}")

    return df_grouped


def load_and_transform(filepath: str) -> pd.DataFrame:
    """
    Load raw Excel file and transform to model format.

    Args:
        filepath: Path to Fact_Raw_New.xlsx

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    logger.info(f"Loading raw data from {filepath}...")
    df_raw = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df_raw)} rows with {len(df_raw.columns)} columns")

    return transform_raw_data(df_raw)


if __name__ == '__main__':
    import sys

    # Default path
    filepath = 'Fact_Raw_New.xlsx'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    df = load_and_transform(filepath)

    print("\nTransformed data sample:")
    print(df.head(20))

    print("\nColumn summary:")
    print(df.info())

    # Save to CSV for inspection
    output_path = 'Fact_Raw_Transformed.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved transformed data to {output_path}")
