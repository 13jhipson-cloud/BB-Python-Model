#!/usr/bin/env python3
"""
Complete Backbook Forecasting Model

This module forecasts loan portfolio performance (collections, GBV, impairment, NBV)
for 12-36 months using historical rate curves and impairment assumptions.

Usage:
    python backbook_forecast.py --fact-raw Fact_Raw_Full.csv --methodology Rate_Methodology.csv

Author: Claude Code
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the backbook forecast model."""

    MAX_MONTHS: int = 12  # Default forecast horizon
    LOOKBACK_PERIODS: int = 6  # Default lookback for CohortAvg
    MOB_THRESHOLD: int = 3  # Minimum MOB for rate calculation

    # Debt Sale Configuration
    # Coverage ratio - percentage of provisions covering debt sale pool
    DS_COVERAGE_RATIO: float = 0.785  # 78.5%
    # Proceeds rate - pence received per £1 of GBV sold
    DS_PROCEEDS_RATE: float = 0.24  # 24p per £1
    # Debt sale months - calendar months when debt sales occur (quarterly)
    DS_MONTHS: List[int] = [3, 6, 9, 12]  # March, June, September, December

    # Rate caps by metric
    RATE_CAPS: Dict[str, Tuple[float, float]] = {
        'Coll_Principal': (-0.15, 0.0),
        'Coll_Interest': (-0.10, 0.0),
        'InterestRevenue': (0.10, 0.50),
        'WO_DebtSold': (0.0, 0.12),
        'WO_Other': (0.0, 0.01),
        'ContraSettlements_Principal': (-0.06, 0.0),
        'ContraSettlements_Interest': (-0.005, 0.0),
        'NewLoanAmount': (0.0, 1.0),
        'Total_Coverage_Ratio': (0.0, 2.50),  # Allow up to 250% coverage (some mature cohorts exceed 100%)
        'Debt_Sale_Coverage_Ratio': (0.50, 1.00),
        'Debt_Sale_Proceeds_Rate': (0.30, 1.00),
    }

    # Valid segments
    SEGMENTS: List[str] = ['NON PRIME', 'NRP-S', 'NRP-M', 'NRP-L', 'PRIME']

    # Metrics for rate calculation
    METRICS: List[str] = [
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ContraSettlements_Principal',
        'ContraSettlements_Interest', 'NewLoanAmount',
        'Total_Coverage_Ratio', 'Debt_Sale_Coverage_Ratio',
        'Debt_Sale_Proceeds_Rate'
    ]

    # Valid rate calculation approaches
    VALID_APPROACHES: List[str] = [
        'CohortAvg', 'CohortTrend', 'DonorCohort',
        'SegMedian', 'Manual', 'Zero'
    ]


# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def parse_date(date_val: Any) -> pd.Timestamp:
    """
    Parse date value to pandas Timestamp.

    Handles both M/D/YYYY and MM/DD/YYYY formats.

    Args:
        date_val: Date value to parse (string, datetime, or Timestamp)

    Returns:
        pd.Timestamp: Parsed date
    """
    if pd.isna(date_val):
        return pd.NaT
    if isinstance(date_val, pd.Timestamp):
        return date_val
    if isinstance(date_val, datetime):
        return pd.Timestamp(date_val)

    # Try parsing as string
    try:
        return pd.to_datetime(date_val, format='%m/%d/%Y')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_val, format='%Y-%m-%d')
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(date_val)
            except Exception:
                logger.warning(f"Could not parse date: {date_val}")
                return pd.NaT


def end_of_month(date: pd.Timestamp) -> pd.Timestamp:
    """
    Get the last day of the month for a given date.

    Args:
        date: Input date

    Returns:
        pd.Timestamp: Last day of the month
    """
    if pd.isna(date):
        return pd.NaT
    return date + pd.offsets.MonthEnd(0)


def clean_cohort(cohort_val: Any) -> str:
    """
    Clean cohort value to string format.

    Args:
        cohort_val: Cohort value (int, float, or string)

    Returns:
        str: Cleaned cohort string (YYYYMM format)
    """
    if pd.isna(cohort_val):
        return ''
    if isinstance(cohort_val, (int, float)):
        return str(int(cohort_val))
    cohort_str = str(cohort_val).replace('.0', '').strip()
    return cohort_str


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        float: Result of division or default
    """
    if pd.isna(denominator) or denominator == 0:
        return default
    if pd.isna(numerator):
        return default
    result = numerator / denominator
    if np.isinf(result) or np.isnan(result):
        return default
    return result


def is_debt_sale_month(date: pd.Timestamp) -> bool:
    """
    Check if a calendar month is a debt sale month.

    Debt sales occur quarterly: March, June, September, December.

    Args:
        date: Calendar date (Timestamp)

    Returns:
        bool: True if this is a debt sale month
    """
    if pd.isna(date):
        return False
    return date.month in Config.DS_MONTHS


# =============================================================================
# SECTION 3: DATA LOADING FUNCTIONS
# =============================================================================

def load_fact_raw(filepath: str) -> pd.DataFrame:
    """
    Load and validate historical loan data.

    Supports both CSV (.csv) and Excel (.xlsx) file formats.
    Automatically maps column names from common variations.

    Args:
        filepath: Path to Fact_Raw file (CSV or Excel)

    Returns:
        pd.DataFrame: Validated fact raw data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading fact raw data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load based on file extension
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(filepath)
        logger.info(f"Loaded {len(df)} rows from Excel file")
    else:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from CSV file")

    # Column name mappings (source -> target)
    # Maps variations found in different data sources to standard names
    column_mappings = {
        'Provision': 'Provision_Balance',
        'DebtSaleProceeds': 'Debt_Sale_Proceeds',
    }

    # Apply column mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            logger.info(f"Renamed column '{old_name}' to '{new_name}'")

    # Required columns (core fields that must exist)
    required_cols = [
        'CalendarMonth', 'Cohort', 'Segment', 'MOB', 'OpeningGBV',
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported', 'DaysInMonth'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse dates (handle both string and datetime formats)
    if not pd.api.types.is_datetime64_any_dtype(df['CalendarMonth']):
        df['CalendarMonth'] = df['CalendarMonth'].apply(parse_date)
    df['CalendarMonth'] = df['CalendarMonth'].apply(end_of_month)

    # Clean cohort (convert to string format YYYYMM)
    df['Cohort'] = df['Cohort'].apply(clean_cohort)

    # Ensure numeric columns
    numeric_cols = [
        'MOB', 'OpeningGBV', 'Coll_Principal', 'Coll_Interest',
        'InterestRevenue', 'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported', 'DaysInMonth'
    ]

    # Optional columns that may or may not exist
    optional_numeric_cols = [
        'NewLoanAmount', 'ContraSettlements_Principal', 'ContraSettlements_Interest'
    ]
    for col in optional_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.info(f"Added missing column {col} with default value 0")

    # Impairment columns (optional, default to 0)
    impairment_cols = [
        'Provision_Balance', 'Debt_Sale_WriteOffs',
        'Debt_Sale_Provision_Release', 'Debt_Sale_Proceeds'
    ]
    for col in impairment_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.info(f"Added missing column {col} with default value 0")

    # Convert all numeric columns
    all_numeric = numeric_cols + optional_numeric_cols + impairment_cols
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Handle provision balance sign convention
    # Some systems store provisions as negative (liability), convert to positive for calculations
    if 'Provision_Balance' in df.columns:
        if df['Provision_Balance'].sum() < 0:
            logger.info("Converting negative provision balances to positive values")
            df['Provision_Balance'] = df['Provision_Balance'].abs()

    # Ensure MOB is integer
    df['MOB'] = df['MOB'].astype(int)

    # Sort data
    df = df.sort_values(['CalendarMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    # Log summary statistics
    logger.info(f"Validated {len(df)} rows with {df['Cohort'].nunique()} cohorts")
    logger.info(f"Segments: {df['Segment'].unique().tolist()}")
    logger.info(f"Date range: {df['CalendarMonth'].min()} to {df['CalendarMonth'].max()}")
    logger.info(f"MOB range: {df['MOB'].min()} to {df['MOB'].max()}")

    return df


def load_rate_methodology(filepath: str) -> pd.DataFrame:
    """
    Load rate calculation control table.

    Args:
        filepath: Path to Rate_Methodology.csv

    Returns:
        pd.DataFrame: Methodology rules
    """
    logger.info(f"Loading rate methodology from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} methodology rules")

    # Fill NaN with "ALL"
    for col in ['Segment', 'Cohort', 'Metric']:
        if col in df.columns:
            df[col] = df[col].fillna('ALL').astype(str).str.strip()

    # Clean cohort
    df['Cohort'] = df['Cohort'].apply(lambda x: clean_cohort(x) if x != 'ALL' else 'ALL')

    # Ensure MOB range columns are integers
    df['MOB_Start'] = pd.to_numeric(df['MOB_Start'], errors='coerce').fillna(0).astype(int)
    df['MOB_End'] = pd.to_numeric(df['MOB_End'], errors='coerce').fillna(999).astype(int)

    # Clean Approach
    df['Approach'] = df['Approach'].astype(str).str.strip()

    # Clean Param1 and Param2
    if 'Param1' in df.columns:
        df['Param1'] = df['Param1'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    else:
        df['Param1'] = None

    if 'Param2' in df.columns:
        df['Param2'] = df['Param2'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    else:
        df['Param2'] = None

    # Validate approaches
    invalid_approaches = df[~df['Approach'].isin(Config.VALID_APPROACHES)]['Approach'].unique()
    if len(invalid_approaches) > 0:
        logger.warning(f"Found invalid approaches: {invalid_approaches}")

    return df


def load_debt_sale_schedule(filepath: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load debt sale assumptions (optional).

    Args:
        filepath: Path to Debt_Sale_Schedule.csv or None

    Returns:
        pd.DataFrame or None: Debt sale schedule
    """
    if filepath is None or not os.path.exists(filepath):
        logger.info("No debt sale schedule loaded")
        return None

    logger.info(f"Loading debt sale schedule from: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} debt sale entries")

    # Parse dates
    df['ForecastMonth'] = df['ForecastMonth'].apply(parse_date)
    df['ForecastMonth'] = df['ForecastMonth'].apply(end_of_month)

    # Clean cohort
    df['Cohort'] = df['Cohort'].apply(clean_cohort)

    # Ensure numeric columns
    for col in ['Debt_Sale_WriteOffs', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.sort_values(['ForecastMonth', 'Segment', 'Cohort']).reset_index(drop=True)

    return df


# =============================================================================
# SECTION 4: CURVES CALCULATION FUNCTIONS
# =============================================================================

def calculate_curves_base(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical rates from actuals.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Base curves with rates by Segment × Cohort × MOB
    """
    logger.info("Calculating base curves...")

    # Group by Segment, Cohort, MOB
    agg_dict = {
        'OpeningGBV': 'sum',
        'NewLoanAmount': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'InterestRevenue': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'ContraSettlements_Principal': 'sum',
        'ContraSettlements_Interest': 'sum',
        'DaysInMonth': 'mean',
        'ClosingGBV_Reported': 'sum',
        'Provision_Balance': 'sum',
        'Debt_Sale_WriteOffs': 'sum',
        'Debt_Sale_Provision_Release': 'sum',
        'Debt_Sale_Proceeds': 'sum',
    }

    curves = fact_raw.groupby(['Segment', 'Cohort', 'MOB']).agg(agg_dict).reset_index()

    # Calculate rates
    curves['NewLoanAmount_Rate'] = curves.apply(
        lambda r: safe_divide(r['NewLoanAmount'], r['OpeningGBV']), axis=1
    )
    curves['Coll_Principal_Rate'] = curves.apply(
        lambda r: safe_divide(r['Coll_Principal'], r['OpeningGBV']), axis=1
    )
    curves['Coll_Interest_Rate'] = curves.apply(
        lambda r: safe_divide(r['Coll_Interest'], r['OpeningGBV']), axis=1
    )
    # Annualize interest revenue rate
    curves['InterestRevenue_Rate'] = curves.apply(
        lambda r: safe_divide(r['InterestRevenue'], r['OpeningGBV']) * safe_divide(365, r['DaysInMonth'], 12),
        axis=1
    )
    curves['WO_DebtSold_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_DebtSold'], r['OpeningGBV']), axis=1
    )
    curves['WO_Other_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_Other'], r['OpeningGBV']), axis=1
    )
    curves['ContraSettlements_Principal_Rate'] = curves.apply(
        lambda r: safe_divide(r['ContraSettlements_Principal'], r['OpeningGBV']), axis=1
    )
    curves['ContraSettlements_Interest_Rate'] = curves.apply(
        lambda r: safe_divide(r['ContraSettlements_Interest'], r['OpeningGBV']), axis=1
    )

    # Calculate coverage ratios
    curves['Total_Coverage_Ratio'] = curves.apply(
        lambda r: safe_divide(r['Provision_Balance'], r['ClosingGBV_Reported']), axis=1
    )
    curves['Debt_Sale_Coverage_Ratio'] = curves.apply(
        lambda r: safe_divide(r['Debt_Sale_Provision_Release'], r['Debt_Sale_WriteOffs']), axis=1
    )
    curves['Debt_Sale_Proceeds_Rate'] = curves.apply(
        lambda r: safe_divide(r['Debt_Sale_Proceeds'], r['Debt_Sale_WriteOffs']), axis=1
    )

    curves = curves.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Calculated curves for {len(curves)} Segment × Cohort × MOB combinations")
    return curves


def extend_curves(curves_base: pd.DataFrame, max_months: int) -> pd.DataFrame:
    """
    Extend curves beyond max observed MOB for forecasting.

    Args:
        curves_base: Base curves with historical rates
        max_months: Number of months to extend

    Returns:
        pd.DataFrame: Extended curves
    """
    logger.info(f"Extending curves for {max_months} months...")

    # Rate columns to extend
    rate_cols = [col for col in curves_base.columns if col.endswith('_Rate')]

    extensions = []

    # Group by Segment and Cohort
    for (segment, cohort), group in curves_base.groupby(['Segment', 'Cohort']):
        max_mob = group['MOB'].max()
        last_row = group[group['MOB'] == max_mob].iloc[0]

        for offset in range(1, max_months + 1):
            new_mob = max_mob + offset
            new_row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': new_mob,
            }
            # Copy rate columns from last MOB
            for col in rate_cols:
                new_row[col] = last_row[col]

            # Copy other columns with defaults
            for col in ['OpeningGBV', 'NewLoanAmount', 'Coll_Principal', 'Coll_Interest',
                        'InterestRevenue', 'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported',
                        'ContraSettlements_Principal', 'ContraSettlements_Interest',
                        'Provision_Balance', 'Debt_Sale_WriteOffs', 'Debt_Sale_Provision_Release',
                        'Debt_Sale_Proceeds']:
                if col in curves_base.columns:
                    new_row[col] = 0.0

            new_row['DaysInMonth'] = 30

            extensions.append(new_row)

    if extensions:
        extensions_df = pd.DataFrame(extensions)
        curves_extended = pd.concat([curves_base, extensions_df], ignore_index=True)
    else:
        curves_extended = curves_base.copy()

    curves_extended = curves_extended.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Extended curves to {len(curves_extended)} rows")
    return curves_extended


# =============================================================================
# SECTION 5: IMPAIRMENT CURVES FUNCTIONS
# =============================================================================

def calculate_impairment_actuals(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate impairment metrics from historical data.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Impairment actuals
    """
    logger.info("Calculating impairment actuals...")

    # Group by Segment, Cohort, CalendarMonth
    agg_dict = {
        'Provision_Balance': 'sum',
        'ClosingGBV_Reported': 'sum',
        'Debt_Sale_WriteOffs': 'sum',
        'Debt_Sale_Provision_Release': 'sum',
        'Debt_Sale_Proceeds': 'sum',
        'WO_Other': 'sum',
        'MOB': 'max',
    }

    impairment = fact_raw.groupby(['Segment', 'Cohort', 'CalendarMonth']).agg(agg_dict).reset_index()

    # Rename for clarity
    impairment.rename(columns={
        'Provision_Balance': 'Total_Provision_Balance',
        'ClosingGBV_Reported': 'Total_ClosingGBV',
    }, inplace=True)

    # Calculate coverage ratio
    impairment['Total_Coverage_Ratio'] = impairment.apply(
        lambda r: safe_divide(r['Total_Provision_Balance'], r['Total_ClosingGBV']), axis=1
    )

    # Calculate debt sale coverage and proceeds rate
    impairment['Debt_Sale_Coverage_Ratio'] = impairment.apply(
        lambda r: safe_divide(r['Debt_Sale_Provision_Release'], r['Debt_Sale_WriteOffs']), axis=1
    )
    impairment['Debt_Sale_Proceeds_Rate'] = impairment.apply(
        lambda r: safe_divide(r['Debt_Sale_Proceeds'], r['Debt_Sale_WriteOffs']), axis=1
    )

    # Sort and calculate provision movement
    impairment = impairment.sort_values(['Segment', 'Cohort', 'CalendarMonth']).reset_index(drop=True)

    impairment['Prior_Provision_Balance'] = impairment.groupby(['Segment', 'Cohort'])['Total_Provision_Balance'].shift(1).fillna(0)
    impairment['Total_Provision_Movement'] = impairment['Total_Provision_Balance'] - impairment['Prior_Provision_Balance']

    # Calculate impairment components
    impairment['Non_DS_Provision_Movement'] = impairment['Total_Provision_Movement'] + impairment['Debt_Sale_Provision_Release']
    impairment['Gross_Impairment_ExcludingDS'] = impairment['Non_DS_Provision_Movement'] + impairment['WO_Other']
    impairment['Debt_Sale_Impact'] = (
        impairment['Debt_Sale_WriteOffs'] +
        impairment['Debt_Sale_Provision_Release'] +
        impairment['Debt_Sale_Proceeds']
    )
    impairment['Net_Impairment'] = impairment['Gross_Impairment_ExcludingDS'] + impairment['Debt_Sale_Impact']

    logger.info(f"Calculated impairment actuals for {len(impairment)} entries")
    return impairment


def calculate_impairment_curves(impairment_actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate impairment rates for forecasting.

    Args:
        impairment_actuals: Impairment actuals data

    Returns:
        pd.DataFrame: Impairment curves with rates
    """
    logger.info("Calculating impairment curves...")

    # Group by Segment, Cohort, MOB
    agg_dict = {
        'Total_Provision_Balance': 'mean',
        'Total_ClosingGBV': 'mean',
        'Total_Coverage_Ratio': 'mean',
        'Debt_Sale_Coverage_Ratio': 'mean',
        'Debt_Sale_Proceeds_Rate': 'mean',
        'WO_Other': 'sum',
    }

    curves = impairment_actuals.groupby(['Segment', 'Cohort', 'MOB']).agg(agg_dict).reset_index()

    # Calculate WO_Other rate
    curves['WO_Other_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_Other'], r['Total_ClosingGBV']), axis=1
    )

    curves = curves.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Calculated impairment curves for {len(curves)} entries")
    return curves


# =============================================================================
# SECTION 6: SEED GENERATION FUNCTIONS
# =============================================================================

def generate_seed_curves(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create forecast starting point from last month of actuals.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Seed with 1 row per Segment × Cohort
    """
    logger.info("Generating seed curves...")

    # Get max calendar month
    max_cal = fact_raw['CalendarMonth'].max()
    logger.info(f"Using last month: {max_cal}")

    # Filter to last month
    last_month = fact_raw[fact_raw['CalendarMonth'] == max_cal].copy()

    # Group by Segment, Cohort
    agg_dict = {
        'ClosingGBV_Reported': 'sum',
        'MOB': 'max',
        'Provision_Balance': 'sum',
    }

    seed = last_month.groupby(['Segment', 'Cohort']).agg(agg_dict).reset_index()

    # Rename columns
    seed.rename(columns={
        'ClosingGBV_Reported': 'BoM',
        'Provision_Balance': 'Prior_Provision_Balance',
    }, inplace=True)

    # MOB for forecast is max MOB + 1
    seed['MOB'] = seed['MOB'] + 1

    # Calculate forecast month (max_cal + 1 month)
    seed['ForecastMonth'] = end_of_month(max_cal + relativedelta(months=1))

    # Filter where BoM > 0
    seed = seed[seed['BoM'] > 0].reset_index(drop=True)

    logger.info(f"Generated seed with {len(seed)} Segment × Cohort combinations")
    return seed


def generate_impairment_seed(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create impairment starting point.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Impairment seed
    """
    logger.info("Generating impairment seed...")

    # Get max calendar month
    max_cal = fact_raw['CalendarMonth'].max()

    # Filter to last month
    last_month = fact_raw[fact_raw['CalendarMonth'] == max_cal].copy()

    # Group by Segment, Cohort
    agg_dict = {
        'Provision_Balance': 'sum',
        'ClosingGBV_Reported': 'sum',
    }

    seed = last_month.groupby(['Segment', 'Cohort']).agg(agg_dict).reset_index()

    # Rename columns
    seed.rename(columns={
        'Provision_Balance': 'Prior_Provision_Balance',
        'ClosingGBV_Reported': 'ClosingGBV',
    }, inplace=True)

    # Calculate forecast month
    seed['ForecastMonth'] = end_of_month(max_cal + relativedelta(months=1))

    logger.info(f"Generated impairment seed with {len(seed)} entries")
    return seed


# =============================================================================
# SECTION 7: METHODOLOGY LOOKUP FUNCTIONS
# =============================================================================

def get_specificity_score(row: pd.Series, segment: str, cohort: str, metric: str, mob: int) -> float:
    """
    Calculate specificity score for a methodology rule.

    Scoring:
    - Exact Segment match: +8 points
    - Exact Cohort match: +4 points
    - Exact Metric match: +2 points
    - Narrower MOB range: +1/(1 + MOB_End - MOB_Start) points (tiebreaker)

    Args:
        row: Methodology rule row
        segment: Target segment
        cohort: Target cohort
        metric: Target metric
        mob: Target MOB

    Returns:
        float: Specificity score
    """
    score = 0.0

    # Segment match
    if row['Segment'] == segment:
        score += 8

    # Cohort match
    if row['Cohort'] == cohort:
        score += 4

    # Metric match
    if row['Metric'] == metric:
        score += 2

    # MOB range width (narrower is better)
    mob_range = row['MOB_End'] - row['MOB_Start']
    score += 1 / (1 + mob_range)

    return score


def get_methodology(methodology_df: pd.DataFrame, segment: str, cohort: str,
                   mob: int, metric: str) -> Dict[str, Any]:
    """
    Find best matching rate calculation rule.

    Args:
        methodology_df: Methodology rules DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric: Target metric

    Returns:
        dict: Best matching rule with Approach, Param1, Param2
    """
    cohort_str = clean_cohort(cohort)

    # Filter matching rules
    mask = (
        ((methodology_df['Segment'] == segment) | (methodology_df['Segment'] == 'ALL')) &
        ((methodology_df['Cohort'] == cohort_str) | (methodology_df['Cohort'] == 'ALL')) &
        ((methodology_df['Metric'] == metric) | (methodology_df['Metric'] == 'ALL')) &
        (methodology_df['MOB_Start'] <= mob) &
        (methodology_df['MOB_End'] >= mob)
    )

    matches = methodology_df[mask].copy()

    if len(matches) == 0:
        return {
            'Approach': 'NoMatch_ERROR',
            'Param1': None,
            'Param2': None
        }

    # Calculate specificity scores
    matches['_score'] = matches.apply(
        lambda r: get_specificity_score(r, segment, cohort_str, metric, mob),
        axis=1
    )

    # Get best match
    best_match = matches.loc[matches['_score'].idxmax()]

    return {
        'Approach': best_match['Approach'],
        'Param1': best_match['Param1'],
        'Param2': best_match['Param2']
    }


# =============================================================================
# SECTION 8: RATE CALCULATION FUNCTIONS
# =============================================================================

def fn_cohort_avg(curves_df: pd.DataFrame, segment: str, cohort: str,
                  mob: int, metric_col: str, lookback: int = 6,
                  exclude_zeros: bool = False) -> Optional[float]:
    """
    Calculate average rate from last N MOBs (post-MOB 3).

    IMPORTANT: Only uses historical data (MOB < forecast MOB), not extended curves.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB (the MOB being forecast)
        metric_col: Column name for metric rate
        lookback: Number of periods to look back
        exclude_zeros: If True, only average non-zero rates (for debt sale metrics)

    Returns:
        float or None: Average rate
    """
    cohort_str = clean_cohort(cohort)

    # Filter data - use MOB < mob to only include HISTORICAL data, not extended curves
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] > Config.MOB_THRESHOLD) &
        (curves_df['MOB'] < mob)  # CHANGED: < instead of <= to exclude forecast MOB
    )

    data = curves_df[mask].sort_values('MOB', ascending=False)

    if len(data) < 2:
        return None

    if metric_col not in data.columns:
        return None

    # For debt sale metrics, only average non-zero rates
    # (zeros just mean no debt sale occurred that month)
    if exclude_zeros:
        non_zero_data = data[data[metric_col] > 0]
        if len(non_zero_data) == 0:
            return None
        # Take last N non-zero values
        non_zero_data = non_zero_data.head(lookback)
        rate = non_zero_data[metric_col].mean()
    else:
        # Take last N rows
        data = data.head(lookback)
        rate = data[metric_col].mean()

    if pd.isna(rate):
        return None

    return float(rate)


def fn_cohort_trend(curves_df: pd.DataFrame, segment: str, cohort: str,
                    mob: int, metric_col: str) -> Optional[float]:
    """
    Linear regression extrapolation on post-MOB 3 data.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Predicted rate
    """
    cohort_str = clean_cohort(cohort)

    # Filter data
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] > Config.MOB_THRESHOLD) &
        (curves_df['MOB'] < mob)
    )

    data = curves_df[mask].copy()

    if len(data) < 2:
        return None

    if metric_col not in data.columns:
        return None

    x = data['MOB'].values
    y = data[metric_col].values

    # Remove NaN values
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return None

    x = x[valid_mask]
    y = y[valid_mask]

    # Linear regression: y = a + b*x
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return None

    b = (n * sum_xy - sum_x * sum_y) / denominator
    a = (sum_y - b * sum_x) / n

    # Predict at target MOB
    predicted = a + b * mob

    if np.isnan(predicted) or np.isinf(predicted):
        return None

    return float(predicted)


def fn_donor_cohort(curves_df: pd.DataFrame, segment: str, donor_cohort: str,
                    mob: int, metric_col: str) -> Optional[float]:
    """
    Copy rate from donor cohort at same MOB.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        donor_cohort: Donor cohort YYYYMM
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Donor rate
    """
    donor_cohort_str = clean_cohort(donor_cohort)

    # Filter data
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == donor_cohort_str) &
        (curves_df['MOB'] == mob)
    )

    data = curves_df[mask]

    if len(data) == 0:
        return None

    if metric_col not in data.columns:
        return None

    rate = data[metric_col].iloc[0]

    if pd.isna(rate):
        return None

    return float(rate)


def fn_seg_median(curves_df: pd.DataFrame, segment: str, mob: int,
                  metric_col: str) -> Optional[float]:
    """
    Median rate across all cohorts in segment at MOB.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Median rate
    """
    # Filter data
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['MOB'] == mob)
    )

    data = curves_df[mask]

    if len(data) == 0:
        return None

    if metric_col not in data.columns:
        return None

    rate = data[metric_col].median()

    if pd.isna(rate):
        return None

    return float(rate)


# =============================================================================
# SECTION 9: RATE APPLICATION FUNCTIONS
# =============================================================================

def apply_approach(curves_df: pd.DataFrame, segment: str, cohort: str,
                   mob: int, metric: str, methodology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate rate using specified approach.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric: Target metric
        methodology: Methodology rule dict with Approach, Param1, Param2

    Returns:
        dict: Rate and ApproachTag
    """
    approach = methodology['Approach']
    param1 = methodology['Param1']

    # Determine the column name for this metric
    # Some metrics (coverage ratios) don't follow the {metric}_Rate pattern
    if metric in ['Total_Coverage_Ratio', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']:
        metric_col = metric  # These are stored directly without _Rate suffix
    else:
        metric_col = f"{metric}_Rate"

    if approach == 'NoMatch_ERROR':
        return {'Rate': 0.0, 'ApproachTag': 'NoMatch_ERROR'}

    elif approach == 'Zero':
        return {'Rate': 0.0, 'ApproachTag': 'Zero'}

    elif approach == 'Manual':
        try:
            if param1 is None or param1 == 'None' or param1 == 'nan':
                return {'Rate': 0.0, 'ApproachTag': 'Manual_InvalidParam_ERROR'}
            rate = float(param1)
            return {'Rate': rate, 'ApproachTag': 'Manual'}
        except (ValueError, TypeError):
            return {'Rate': 0.0, 'ApproachTag': 'Manual_InvalidParam_ERROR'}

    elif approach == 'CohortAvg':
        try:
            lookback = int(float(param1)) if param1 and param1 != 'None' else Config.LOOKBACK_PERIODS
        except (ValueError, TypeError):
            lookback = Config.LOOKBACK_PERIODS

        # For debt sale metrics, only average non-zero rates
        # (zeros just mean no debt sale occurred that month, not that the rate is 0)
        exclude_zeros = metric in ['WO_DebtSold', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']

        rate = fn_cohort_avg(curves_df, segment, cohort, mob, metric_col, lookback, exclude_zeros)
        if rate is not None:
            tag = 'CohortAvg_NonZero' if exclude_zeros else 'CohortAvg'
            return {'Rate': rate, 'ApproachTag': tag}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'CohortAvg_NoData_ERROR'}

    elif approach == 'CohortTrend':
        rate = fn_cohort_trend(curves_df, segment, cohort, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': 'CohortTrend'}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'CohortTrend_NoData_ERROR'}

    elif approach == 'SegMedian':
        rate = fn_seg_median(curves_df, segment, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': 'SegMedian'}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'SegMedian_NoData_ERROR'}

    elif approach == 'DonorCohort':
        if param1 is None or param1 == 'None':
            return {'Rate': 0.0, 'ApproachTag': 'DonorCohort_NoParam_ERROR'}

        donor = clean_cohort(param1)
        rate = fn_donor_cohort(curves_df, segment, donor, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': f'DonorCohort:{donor}'}
        else:
            return {'Rate': 0.0, 'ApproachTag': f'DonorCohort_NoData_ERROR:{donor}'}

    else:
        return {'Rate': 0.0, 'ApproachTag': f'UnknownApproach_ERROR:{approach}'}


def apply_rate_cap(rate: float, metric: str, approach_tag: str) -> float:
    """
    Cap rates to reasonable ranges.

    Args:
        rate: Input rate
        metric: Metric name
        approach_tag: Approach tag (caps bypassed for Manual and ERROR)

    Returns:
        float: Capped rate
    """
    if rate is None or pd.isna(rate):
        return 0.0

    # Don't cap Manual overrides or errors
    if 'Manual' in approach_tag or 'ERROR' in approach_tag:
        return rate

    # Apply caps
    if metric in Config.RATE_CAPS:
        min_cap, max_cap = Config.RATE_CAPS[metric]
        return max(min_cap, min(max_cap, rate))

    return rate


# =============================================================================
# SECTION 10: RATE LOOKUP BUILDER
# =============================================================================

def build_rate_lookup(seed: pd.DataFrame, curves: pd.DataFrame,
                      methodology: pd.DataFrame, max_months: int) -> pd.DataFrame:
    """
    Build rate lookup table for forecast.

    Args:
        seed: Seed curves
        curves: Extended curves
        methodology: Methodology rules
        max_months: Forecast horizon

    Returns:
        pd.DataFrame: Rate lookup table
    """
    logger.info("Building rate lookup...")

    # Metrics to calculate rates for
    rate_metrics = [
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ContraSettlements_Principal',
        'ContraSettlements_Interest', 'NewLoanAmount'
    ]

    lookups = []

    for _, seed_row in seed.iterrows():
        segment = seed_row['Segment']
        cohort = seed_row['Cohort']
        start_mob = seed_row['MOB']

        for month_offset in range(max_months):
            mob = start_mob + month_offset

            row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
            }

            for metric in rate_metrics:
                # Get methodology
                meth = get_methodology(methodology, segment, cohort, mob, metric)

                # Apply approach
                result = apply_approach(curves, segment, cohort, mob, metric, meth)

                # Apply cap
                capped_rate = apply_rate_cap(result['Rate'], metric, result['ApproachTag'])

                row[f'{metric}_Rate'] = capped_rate
                row[f'{metric}_Approach'] = result['ApproachTag']

            lookups.append(row)

    lookup_df = pd.DataFrame(lookups)
    logger.info(f"Built rate lookup with {len(lookup_df)} entries")

    return lookup_df


def build_impairment_lookup(seed: pd.DataFrame, impairment_curves: pd.DataFrame,
                            methodology: pd.DataFrame, max_months: int,
                            debt_sale_schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build impairment lookup table for forecast.

    Args:
        seed: Seed curves
        impairment_curves: Impairment curves
        methodology: Methodology rules
        max_months: Forecast horizon
        debt_sale_schedule: Optional debt sale schedule

    Returns:
        pd.DataFrame: Impairment lookup table
    """
    logger.info("Building impairment lookup...")

    impairment_metrics = [
        'Total_Coverage_Ratio', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate'
    ]

    # Get start forecast month from seed
    start_forecast_month = seed['ForecastMonth'].iloc[0]

    lookups = []

    for _, seed_row in seed.iterrows():
        segment = seed_row['Segment']
        cohort = seed_row['Cohort']
        start_mob = seed_row['MOB']

        for month_offset in range(max_months):
            mob = start_mob + month_offset
            forecast_month = end_of_month(start_forecast_month + relativedelta(months=month_offset))

            row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
                'ForecastMonth': forecast_month,
            }

            # Check if this is a debt sale month
            debt_sale_wo = 0.0
            if debt_sale_schedule is not None:
                ds_mask = (
                    (debt_sale_schedule['ForecastMonth'] == forecast_month) &
                    (debt_sale_schedule['Segment'] == segment) &
                    (debt_sale_schedule['Cohort'] == cohort)
                )
                if ds_mask.any():
                    ds_row = debt_sale_schedule[ds_mask].iloc[0]
                    debt_sale_wo = ds_row.get('Debt_Sale_WriteOffs', 0.0)
                    row['Debt_Sale_Coverage_Ratio'] = ds_row.get('Debt_Sale_Coverage_Ratio', 0.85)
                    row['Debt_Sale_Proceeds_Rate'] = ds_row.get('Debt_Sale_Proceeds_Rate', 0.90)

            row['Debt_Sale_WriteOffs'] = debt_sale_wo

            # Get coverage ratio from methodology
            meth = get_methodology(methodology, segment, cohort, mob, 'Total_Coverage_Ratio')
            result = apply_approach(impairment_curves, segment, cohort, mob, 'Total_Coverage_Ratio', meth)

            if result['Rate'] == 0.0 and 'ERROR' in result['ApproachTag']:
                # Fallback to curves if available
                mask = (
                    (impairment_curves['Segment'] == segment) &
                    (impairment_curves['Cohort'] == cohort)
                )
                if mask.any():
                    avg_coverage = impairment_curves[mask]['Total_Coverage_Ratio'].mean()
                    if not pd.isna(avg_coverage):
                        result['Rate'] = avg_coverage

            capped_rate = apply_rate_cap(result['Rate'], 'Total_Coverage_Ratio', result['ApproachTag'])
            row['Total_Coverage_Ratio'] = capped_rate
            row['Total_Coverage_Approach'] = result['ApproachTag']

            # Set defaults for debt sale ratios if not already set
            if 'Debt_Sale_Coverage_Ratio' not in row:
                row['Debt_Sale_Coverage_Ratio'] = 0.85
            if 'Debt_Sale_Proceeds_Rate' not in row:
                row['Debt_Sale_Proceeds_Rate'] = 0.90

            lookups.append(row)

    lookup_df = pd.DataFrame(lookups)
    logger.info(f"Built impairment lookup with {len(lookup_df)} entries")

    return lookup_df


# =============================================================================
# SECTION 11: FORECAST ENGINE FUNCTIONS
# =============================================================================

def run_one_step(seed_table: pd.DataFrame, rate_lookup: pd.DataFrame,
                 impairment_lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute one month of forecast.

    Args:
        seed_table: Current seed with BoM, MOB, ForecastMonth
        rate_lookup: Rate lookup table
        impairment_lookup: Impairment lookup table

    Returns:
        tuple: (step_output_df, next_seed_df)
    """
    outputs = []
    next_seeds = []

    for _, seed_row in seed_table.iterrows():
        segment = seed_row['Segment']
        cohort = seed_row['Cohort']
        mob = seed_row['MOB']
        bom = seed_row['BoM']
        forecast_month = seed_row['ForecastMonth']
        prior_provision = seed_row.get('Prior_Provision_Balance', 0.0)

        # Get rates
        rate_mask = (
            (rate_lookup['Segment'] == segment) &
            (rate_lookup['Cohort'] == cohort) &
            (rate_lookup['MOB'] == mob)
        )

        if not rate_mask.any():
            continue

        rates = rate_lookup[rate_mask].iloc[0]

        # Get impairment rates
        imp_mask = (
            (impairment_lookup['Segment'] == segment) &
            (impairment_lookup['Cohort'] == cohort) &
            (impairment_lookup['MOB'] == mob)
        )

        if not imp_mask.any():
            continue

        imp_rates = impairment_lookup[imp_mask].iloc[0]

        # Calculate amounts
        opening_gbv = bom

        new_loan_amount = opening_gbv * rates.get('NewLoanAmount_Rate', 0.0)
        coll_principal = opening_gbv * rates.get('Coll_Principal_Rate', 0.0)
        coll_interest = opening_gbv * rates.get('Coll_Interest_Rate', 0.0)
        interest_revenue = opening_gbv * rates.get('InterestRevenue_Rate', 0.0) / 12  # Monthly

        # WO_DebtSold only occurs in debt sale months (Mar, Jun, Sep, Dec)
        if is_debt_sale_month(forecast_month):
            wo_debt_sold = opening_gbv * rates.get('WO_DebtSold_Rate', 0.0)
        else:
            wo_debt_sold = 0.0

        wo_other = opening_gbv * rates.get('WO_Other_Rate', 0.0)
        contra_principal = opening_gbv * rates.get('ContraSettlements_Principal_Rate', 0.0)
        contra_interest = opening_gbv * rates.get('ContraSettlements_Interest_Rate', 0.0)

        # Calculate closing GBV
        closing_gbv = (
            opening_gbv +
            interest_revenue -
            abs(coll_principal) -
            abs(coll_interest) -
            wo_debt_sold -
            wo_other
        )

        # Ensure non-negative
        closing_gbv = max(0.0, closing_gbv)

        # =======================================================================
        # DEBT SALE AND IMPAIRMENT CALCULATION
        # =======================================================================
        # User's expected calculation:
        # - WO_DebtSold (forecast from rates) IS the Debt Sale WriteOffs
        # - DS Coverage Ratio = 78.5% fixed for all
        # - DS provision for DS pool = DS coverage ratio × WO_DebtSold
        # - Core provision = Prior provision - DS provision for DS pool
        # - Core GBV = Prior closing GBV - DS WriteOffs (i.e., OpeningGBV - WO_DebtSold)
        # - Core coverage ratio = Core provision / Core GBV
        # =======================================================================

        # Debt sale writeoffs = WO_DebtSold forecast from rates
        debt_sale_wo = wo_debt_sold  # Use WO_DebtSold from rates as debt sale writeoffs
        ds_coverage_ratio = Config.DS_COVERAGE_RATIO  # Fixed 78.5%
        ds_proceeds_rate = Config.DS_PROCEEDS_RATE  # Fixed 24p per £1 of GBV sold

        # Calculate DS provision for DS pool
        ds_provision_for_pool = ds_coverage_ratio * debt_sale_wo

        # Calculate core values (after removing debt sale portion)
        core_provision = prior_provision - ds_provision_for_pool
        core_gbv = opening_gbv - debt_sale_wo  # Prior closing = current opening

        # Calculate core coverage ratio (provision on remaining "good" loans)
        core_coverage_ratio = safe_divide(core_provision, core_gbv, default=0.0)

        # Calculate DS proceeds
        ds_proceeds = ds_proceeds_rate * debt_sale_wo

        # Total provision balance (based on methodology coverage ratio applied to closing GBV)
        total_coverage_ratio = imp_rates.get('Total_Coverage_Ratio', 0.12)
        total_provision_balance = closing_gbv * total_coverage_ratio
        total_provision_movement = total_provision_balance - prior_provision

        # Provision release from debt sale (provision that was covering sold loans)
        ds_provision_release = ds_provision_for_pool

        # Calculate net impairment components
        non_ds_provision_movement = total_provision_movement + ds_provision_release
        gross_impairment_excl_ds = non_ds_provision_movement + wo_other
        debt_sale_impact = debt_sale_wo + ds_provision_release + ds_proceeds
        net_impairment = gross_impairment_excl_ds + debt_sale_impact

        # Calculate closing NBV
        closing_nbv = closing_gbv - net_impairment

        # Build output row
        output_row = {
            'ForecastMonth': forecast_month,
            'Segment': segment,
            'Cohort': cohort,
            'MOB': mob,
            'OpeningGBV': round(opening_gbv, 2),

            # Rates
            'Coll_Principal_Rate': rates.get('Coll_Principal_Rate', 0.0),
            'Coll_Principal_Approach': rates.get('Coll_Principal_Approach', ''),
            'Coll_Interest_Rate': rates.get('Coll_Interest_Rate', 0.0),
            'Coll_Interest_Approach': rates.get('Coll_Interest_Approach', ''),
            'InterestRevenue_Rate': rates.get('InterestRevenue_Rate', 0.0),
            'InterestRevenue_Approach': rates.get('InterestRevenue_Approach', ''),
            'WO_DebtSold_Rate': rates.get('WO_DebtSold_Rate', 0.0),
            'WO_DebtSold_Approach': rates.get('WO_DebtSold_Approach', ''),
            'WO_Other_Rate': rates.get('WO_Other_Rate', 0.0),
            'WO_Other_Approach': rates.get('WO_Other_Approach', ''),
            'NewLoanAmount_Rate': rates.get('NewLoanAmount_Rate', 0.0),
            'NewLoanAmount_Approach': rates.get('NewLoanAmount_Approach', ''),
            'ContraSettlements_Principal_Rate': rates.get('ContraSettlements_Principal_Rate', 0.0),
            'ContraSettlements_Principal_Approach': rates.get('ContraSettlements_Principal_Approach', ''),
            'ContraSettlements_Interest_Rate': rates.get('ContraSettlements_Interest_Rate', 0.0),
            'ContraSettlements_Interest_Approach': rates.get('ContraSettlements_Interest_Approach', ''),

            # Amounts
            'NewLoanAmount': round(new_loan_amount, 2),
            'Coll_Principal': round(coll_principal, 2),
            'Coll_Interest': round(coll_interest, 2),
            'InterestRevenue': round(interest_revenue, 2),
            'WO_DebtSold': round(wo_debt_sold, 2),
            'WO_Other': round(wo_other, 2),
            'ContraSettlements_Principal': round(contra_principal, 2),
            'ContraSettlements_Interest': round(contra_interest, 2),

            # GBV
            'ClosingGBV': round(closing_gbv, 2),

            # Impairment
            'Total_Coverage_Ratio': round(total_coverage_ratio, 6),
            'Total_Coverage_Approach': imp_rates.get('Total_Coverage_Approach', ''),
            'Total_Provision_Balance': round(total_provision_balance, 2),
            'Prior_Provision_Balance': round(prior_provision, 2),
            'Total_Provision_Movement': round(total_provision_movement, 2),

            # Debt Sale - only occurs in debt sale months (Mar, Jun, Sep, Dec)
            'Is_Debt_Sale_Month': is_debt_sale_month(forecast_month),
            'Debt_Sale_WriteOffs': round(debt_sale_wo, 2),
            'Debt_Sale_Coverage_Ratio': round(ds_coverage_ratio, 6),
            'Debt_Sale_Proceeds_Rate': round(ds_proceeds_rate, 6),
            'DS_Provision_For_Pool': round(ds_provision_for_pool, 2),
            'Debt_Sale_Provision_Release': round(ds_provision_release, 2),
            'Debt_Sale_Proceeds': round(ds_proceeds, 2),

            # Core values (after removing debt sale portion)
            'Core_Provision': round(core_provision, 2),
            'Core_GBV': round(core_gbv, 2),
            'Core_Coverage_Ratio': round(core_coverage_ratio, 6),

            # Net impairment components
            'Non_DS_Provision_Movement': round(non_ds_provision_movement, 2),
            'Gross_Impairment_ExcludingDS': round(gross_impairment_excl_ds, 2),
            'Debt_Sale_Impact': round(debt_sale_impact, 2),
            'Net_Impairment': round(net_impairment, 2),

            # NBV
            'ClosingNBV': round(closing_nbv, 2),
        }

        outputs.append(output_row)

        # Prepare next seed (if closing GBV > 0)
        if closing_gbv > 0:
            next_forecast_month = end_of_month(forecast_month + relativedelta(months=1))
            next_seeds.append({
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob + 1,
                'BoM': closing_gbv,
                'ForecastMonth': next_forecast_month,
                'Prior_Provision_Balance': total_provision_balance,
            })

    step_output = pd.DataFrame(outputs)
    next_seed = pd.DataFrame(next_seeds)

    return step_output, next_seed


def run_forecast(seed: pd.DataFrame, rate_lookup: pd.DataFrame,
                 impairment_lookup: pd.DataFrame, max_months: int) -> pd.DataFrame:
    """
    Run complete forecast loop.

    Args:
        seed: Starting seed
        rate_lookup: Rate lookup table
        impairment_lookup: Impairment lookup table
        max_months: Forecast horizon

    Returns:
        pd.DataFrame: Complete forecast
    """
    logger.info(f"Running forecast for {max_months} months...")

    all_outputs = []
    current_seed = seed.copy()

    for month in range(max_months):
        if len(current_seed) == 0:
            logger.info(f"No more active cohorts at month {month + 1}")
            break

        logger.info(f"Forecasting month {month + 1} with {len(current_seed)} cohorts")

        step_output, next_seed = run_one_step(current_seed, rate_lookup, impairment_lookup)

        if len(step_output) > 0:
            all_outputs.append(step_output)

        current_seed = next_seed

    if not all_outputs:
        logger.warning("No forecast output generated")
        return pd.DataFrame()

    forecast = pd.concat(all_outputs, ignore_index=True)
    forecast = forecast.sort_values(['ForecastMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Forecast complete with {len(forecast)} rows")
    return forecast


# =============================================================================
# SECTION 12: OUTPUT GENERATION FUNCTIONS
# =============================================================================

def generate_summary_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level summary for Excel.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Summary by ForecastMonth and Segment
    """
    logger.info("Generating summary output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    agg_dict = {
        'OpeningGBV': 'sum',
        'InterestRevenue': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'ClosingGBV': 'sum',
        'Total_Provision_Balance': 'sum',
        'Net_Impairment': 'sum',
        'ClosingNBV': 'sum',
    }

    summary = forecast.groupby(['ForecastMonth', 'Segment']).agg(agg_dict).reset_index()

    # Calculate weighted coverage ratio
    summary['Total_Coverage_Ratio'] = summary.apply(
        lambda r: safe_divide(r['Total_Provision_Balance'], r['ClosingGBV']), axis=1
    )

    # Select and order columns
    columns = [
        'ForecastMonth', 'Segment', 'OpeningGBV', 'InterestRevenue',
        'Coll_Principal', 'Coll_Interest', 'WO_DebtSold', 'WO_Other',
        'ClosingGBV', 'Total_Coverage_Ratio', 'Net_Impairment', 'ClosingNBV'
    ]

    summary = summary[columns].sort_values(['ForecastMonth', 'Segment']).reset_index(drop=True)

    # Round numeric columns
    for col in summary.columns:
        if col not in ['ForecastMonth', 'Segment']:
            summary[col] = summary[col].round(2)

    logger.info(f"Generated summary with {len(summary)} rows")
    return summary


def generate_details_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete forecast for Excel.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Detailed forecast
    """
    logger.info("Generating details output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    details = forecast.copy()

    # Format dates
    details['ForecastMonth'] = pd.to_datetime(details['ForecastMonth']).dt.strftime('%Y-%m-%d')

    details = details.sort_values(['ForecastMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Generated details with {len(details)} rows")
    return details


def generate_impairment_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create impairment-specific analysis.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Impairment analysis
    """
    logger.info("Generating impairment output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    columns = [
        'ForecastMonth', 'Segment', 'Cohort', 'MOB', 'OpeningGBV', 'ClosingGBV',
        'Total_Coverage_Ratio', 'Total_Provision_Balance', 'Prior_Provision_Balance',
        'Total_Provision_Movement',
        # Debt Sale metrics
        'WO_DebtSold', 'Debt_Sale_WriteOffs', 'Debt_Sale_Coverage_Ratio',
        'DS_Provision_For_Pool', 'Debt_Sale_Provision_Release', 'Debt_Sale_Proceeds',
        # Core values (after debt sale)
        'Core_Provision', 'Core_GBV', 'Core_Coverage_Ratio',
        # Net impairment components
        'Non_DS_Provision_Movement', 'Gross_Impairment_ExcludingDS',
        'Debt_Sale_Impact', 'Net_Impairment'
    ]

    impairment = forecast[columns].copy()
    impairment['ForecastMonth'] = pd.to_datetime(impairment['ForecastMonth']).dt.strftime('%Y-%m-%d')
    impairment = impairment.sort_values(['ForecastMonth', 'Segment', 'Cohort']).reset_index(drop=True)

    logger.info(f"Generated impairment output with {len(impairment)} rows")
    return impairment


def generate_validation_output(forecast: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation checks.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        tuple: (reconciliation_df, validation_checks_df)
    """
    logger.info("Generating validation output...")

    if len(forecast) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Reconciliation check
    recon = forecast.copy()

    # GBV reconciliation
    recon['ClosingGBV_Calculated'] = (
        recon['OpeningGBV'] +
        recon['InterestRevenue'] -
        abs(recon['Coll_Principal']) -
        abs(recon['Coll_Interest']) -
        recon['WO_DebtSold'] -
        recon['WO_Other']
    ).round(2)

    recon['GBV_Variance'] = (recon['ClosingGBV_Calculated'] - recon['ClosingGBV']).abs().round(2)
    # Use tolerance of 1.0 to account for floating point rounding on large numbers
    recon['GBV_Status'] = recon['GBV_Variance'].apply(lambda x: 'PASS' if x < 1.0 else 'FAIL')

    # NBV reconciliation
    recon['ClosingNBV_Calculated'] = (recon['ClosingGBV'] - recon['Net_Impairment']).round(2)
    recon['NBV_Variance'] = (recon['ClosingNBV_Calculated'] - recon['ClosingNBV']).abs().round(2)
    recon['NBV_Status'] = recon['NBV_Variance'].apply(lambda x: 'PASS' if x < 1.0 else 'FAIL')

    # Select reconciliation columns
    recon_cols = [
        'ForecastMonth', 'Segment', 'Cohort', 'OpeningGBV', 'InterestRevenue',
        'Coll_Principal', 'Coll_Interest', 'WO_DebtSold', 'WO_Other',
        'ClosingGBV_Calculated', 'ClosingGBV', 'GBV_Variance', 'GBV_Status',
        'Net_Impairment', 'ClosingNBV_Calculated', 'ClosingNBV', 'NBV_Variance', 'NBV_Status'
    ]

    reconciliation = recon[recon_cols].copy()
    reconciliation['ForecastMonth'] = pd.to_datetime(reconciliation['ForecastMonth']).dt.strftime('%Y-%m-%d')

    # Validation checks summary
    total_rows = len(forecast)

    checks = [
        {
            'Check': 'GBV_Reconciliation',
            'Total_Rows': total_rows,
            'Passed': (recon['GBV_Status'] == 'PASS').sum(),
            'Failed': (recon['GBV_Status'] == 'FAIL').sum(),
        },
        {
            'Check': 'NBV_Reconciliation',
            'Total_Rows': total_rows,
            'Passed': (recon['NBV_Status'] == 'PASS').sum(),
            'Failed': (recon['NBV_Status'] == 'FAIL').sum(),
        },
        {
            'Check': 'No_NaN_Values',
            'Total_Rows': total_rows,
            'Passed': total_rows - forecast[['OpeningGBV', 'ClosingGBV', 'ClosingNBV']].isna().any(axis=1).sum(),
            'Failed': forecast[['OpeningGBV', 'ClosingGBV', 'ClosingNBV']].isna().any(axis=1).sum(),
        },
        {
            'Check': 'No_Infinite_Values',
            'Total_Rows': total_rows,
            'Passed': total_rows - np.isinf(forecast.select_dtypes(include=[np.number])).any(axis=1).sum(),
            'Failed': np.isinf(forecast.select_dtypes(include=[np.number])).any(axis=1).sum(),
        },
        {
            'Check': 'Coverage_Ratio_Range',
            'Total_Rows': total_rows,
            # Allow coverage ratios between 0 and 1.0 (0% to 100%)
            'Passed': ((forecast['Total_Coverage_Ratio'] >= 0.0) & (forecast['Total_Coverage_Ratio'] <= 1.0)).sum(),
            'Failed': ((forecast['Total_Coverage_Ratio'] < 0.0) | (forecast['Total_Coverage_Ratio'] > 1.0)).sum(),
        },
    ]

    validation_df = pd.DataFrame(checks)
    validation_df['Pass_Rate'] = (validation_df['Passed'] / validation_df['Total_Rows'] * 100).round(1).astype(str) + '%'
    validation_df['Status'] = validation_df.apply(
        lambda r: 'PASS' if r['Failed'] == 0 else 'FAIL', axis=1
    )

    # Overall status
    overall_passed = validation_df['Passed'].sum()
    overall_total = validation_df['Total_Rows'].sum()
    overall_failed = validation_df['Failed'].sum()
    overall_status = 'PASS' if overall_failed == 0 else 'FAIL'

    validation_df = pd.concat([
        validation_df,
        pd.DataFrame([{
            'Check': 'Overall',
            'Total_Rows': overall_total,
            'Passed': overall_passed,
            'Failed': overall_failed,
            'Pass_Rate': f"{overall_passed / overall_total * 100:.1f}%" if overall_total > 0 else '0%',
            'Status': overall_status,
        }])
    ], ignore_index=True)

    logger.info(f"Generated validation output - Overall status: {overall_status}")
    return reconciliation, validation_df


def export_to_excel(summary: pd.DataFrame, details: pd.DataFrame,
                    impairment: pd.DataFrame, reconciliation: pd.DataFrame,
                    validation: pd.DataFrame, output_dir: str) -> None:
    """
    Write all outputs to Excel workbooks.

    Args:
        summary: Summary DataFrame
        details: Details DataFrame
        impairment: Impairment DataFrame
        reconciliation: Reconciliation DataFrame
        validation: Validation checks DataFrame
        output_dir: Output directory path
    """
    logger.info(f"Exporting to Excel in: {output_dir}")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Export Forecast_Summary.xlsx
    summary_path = os.path.join(output_dir, 'Forecast_Summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
    logger.info(f"Created: {summary_path}")

    # Export Forecast_Details.xlsx
    details_path = os.path.join(output_dir, 'Forecast_Details.xlsx')
    with pd.ExcelWriter(details_path, engine='openpyxl') as writer:
        details.to_excel(writer, sheet_name='All_Cohorts', index=False)
    logger.info(f"Created: {details_path}")

    # Export Impairment_Analysis.xlsx
    impairment_path = os.path.join(output_dir, 'Impairment_Analysis.xlsx')
    with pd.ExcelWriter(impairment_path, engine='openpyxl') as writer:
        impairment.to_excel(writer, sheet_name='Impairment_Detail', index=False)

        # Coverage ratios sheet
        if len(impairment) > 0:
            coverage_cols = ['Segment', 'Cohort', 'MOB', 'Total_Coverage_Ratio',
                           'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']
            coverage_cols = [c for c in coverage_cols if c in impairment.columns]
            coverage = impairment[coverage_cols].drop_duplicates()
            coverage.to_excel(writer, sheet_name='Coverage_Ratios', index=False)
    logger.info(f"Created: {impairment_path}")

    # Export Validation_Report.xlsx
    validation_path = os.path.join(output_dir, 'Validation_Report.xlsx')
    with pd.ExcelWriter(validation_path, engine='openpyxl') as writer:
        reconciliation.to_excel(writer, sheet_name='Reconciliation', index=False)
        validation.to_excel(writer, sheet_name='Validation_Checks', index=False)
    logger.info(f"Created: {validation_path}")

    logger.info("Excel export complete")


# =============================================================================
# SECTION 13: MAIN ORCHESTRATION
# =============================================================================

def run_backbook_forecast(fact_raw_path: str, methodology_path: str,
                          debt_sale_path: Optional[str], output_dir: str,
                          max_months: int) -> pd.DataFrame:
    """
    Orchestrate entire forecast process.

    Args:
        fact_raw_path: Path to Fact_Raw_Full.csv
        methodology_path: Path to Rate_Methodology.csv
        debt_sale_path: Path to Debt_Sale_Schedule.csv or None
        output_dir: Output directory
        max_months: Forecast horizon

    Returns:
        pd.DataFrame: Complete forecast
    """
    logger.info("=" * 60)
    logger.info("Starting Backbook Forecast")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        # 1. Load data
        logger.info("\n[Step 1/8] Loading data...")
        fact_raw = load_fact_raw(fact_raw_path)
        methodology = load_rate_methodology(methodology_path)
        debt_sale_schedule = load_debt_sale_schedule(debt_sale_path)

        # 2. Calculate curves
        logger.info("\n[Step 2/8] Calculating curves...")
        curves_base = calculate_curves_base(fact_raw)
        curves_extended = extend_curves(curves_base, max_months)

        # 3. Calculate impairment curves
        logger.info("\n[Step 3/8] Calculating impairment curves...")
        impairment_actuals = calculate_impairment_actuals(fact_raw)
        impairment_curves = calculate_impairment_curves(impairment_actuals)

        # 4. Generate seeds
        logger.info("\n[Step 4/8] Generating seeds...")
        seed = generate_seed_curves(fact_raw)

        # 5. Build rate lookups
        logger.info("\n[Step 5/8] Building rate lookups...")
        rate_lookup = build_rate_lookup(seed, curves_extended, methodology, max_months)
        impairment_lookup = build_impairment_lookup(
            seed, impairment_curves, methodology, max_months, debt_sale_schedule
        )

        # 6. Run forecast
        logger.info("\n[Step 6/8] Running forecast...")
        forecast = run_forecast(seed, rate_lookup, impairment_lookup, max_months)

        if len(forecast) == 0:
            logger.error("No forecast data generated")
            return pd.DataFrame()

        # 7. Generate outputs
        logger.info("\n[Step 7/8] Generating outputs...")
        summary = generate_summary_output(forecast)
        details = generate_details_output(forecast)
        impairment = generate_impairment_output(forecast)
        reconciliation, validation = generate_validation_output(forecast)

        # 8. Export to Excel
        logger.info("\n[Step 8/8] Exporting to Excel...")
        export_to_excel(summary, details, impairment, reconciliation, validation, output_dir)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info(f"Forecast complete in {elapsed:.2f} seconds")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)

        # Print validation summary
        if len(validation) > 0:
            overall = validation[validation['Check'] == 'Overall'].iloc[0]
            logger.info(f"\nValidation Summary: {overall['Status']}")
            logger.info(f"  Total checks: {overall['Total_Rows']}")
            logger.info(f"  Passed: {overall['Passed']}")
            logger.info(f"  Failed: {overall['Failed']}")

        return forecast

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Backbook Forecasting Model - Calculate loan portfolio performance forecasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backbook_forecast.py --fact-raw Fact_Raw_Full.csv --methodology Rate_Methodology.csv
  python backbook_forecast.py --fact-raw data/Fact_Raw_Full.csv --methodology data/Rate_Methodology.csv --months 24 --output results/
        """
    )

    parser.add_argument(
        '--fact-raw', '-f',
        required=True,
        help='Path to Fact_Raw_Full.csv (historical loan data)'
    )

    parser.add_argument(
        '--methodology', '-m',
        required=True,
        help='Path to Rate_Methodology.csv (rate calculation rules)'
    )

    parser.add_argument(
        '--debt-sale', '-d',
        required=False,
        default=None,
        help='Path to Debt_Sale_Schedule.csv (optional debt sale assumptions)'
    )

    parser.add_argument(
        '--output', '-o',
        required=False,
        default='output',
        help='Output directory (default: output/)'
    )

    parser.add_argument(
        '--months', '-n',
        required=False,
        type=int,
        default=12,
        help='Forecast horizon in months (default: 12)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_backbook_forecast(
        fact_raw_path=args.fact_raw,
        methodology_path=args.methodology,
        debt_sale_path=args.debt_sale,
        output_dir=args.output,
        max_months=args.months
    )


if __name__ == '__main__':
    main()
