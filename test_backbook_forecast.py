#!/usr/bin/env python3
"""
Unit and Integration Tests for Backbook Forecasting Model

Run tests with:
    python -m pytest test_backbook_forecast.py -v
    python -m pytest test_backbook_forecast.py -v --tb=short
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Import the backbook_forecast module
from backbook_forecast import (
    Config,
    parse_date,
    end_of_month,
    clean_cohort,
    safe_divide,
    load_fact_raw,
    load_rate_methodology,
    load_debt_sale_schedule,
    calculate_curves_base,
    extend_curves,
    calculate_impairment_actuals,
    calculate_impairment_curves,
    generate_seed_curves,
    generate_impairment_seed,
    get_specificity_score,
    get_methodology,
    fn_cohort_avg,
    fn_cohort_trend,
    fn_donor_cohort,
    fn_seg_median,
    apply_approach,
    apply_rate_cap,
    build_rate_lookup,
    build_impairment_lookup,
    run_one_step,
    run_forecast,
    generate_summary_output,
    generate_details_output,
    generate_impairment_output,
    generate_validation_output,
    run_backbook_forecast,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_data_dir():
    """Return path to sample data directory."""
    return os.path.join(os.path.dirname(__file__), 'sample_data')


@pytest.fixture
def fact_raw_path(sample_data_dir):
    """Return path to sample Fact_Raw_Full.csv."""
    return os.path.join(sample_data_dir, 'Fact_Raw_Full.csv')


@pytest.fixture
def methodology_path(sample_data_dir):
    """Return path to sample Rate_Methodology.csv."""
    return os.path.join(sample_data_dir, 'Rate_Methodology.csv')


@pytest.fixture
def debt_sale_path(sample_data_dir):
    """Return path to sample Debt_Sale_Schedule.csv."""
    return os.path.join(sample_data_dir, 'Debt_Sale_Schedule.csv')


@pytest.fixture
def temp_output_dir():
    """Create and return a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def fact_raw_df(fact_raw_path):
    """Load and return sample fact raw DataFrame."""
    return load_fact_raw(fact_raw_path)


@pytest.fixture
def methodology_df(methodology_path):
    """Load and return sample methodology DataFrame."""
    return load_rate_methodology(methodology_path)


@pytest.fixture
def curves_base_df(fact_raw_df):
    """Calculate and return base curves DataFrame."""
    return calculate_curves_base(fact_raw_df)


# =============================================================================
# UNIT TESTS: HELPER FUNCTIONS
# =============================================================================

class TestHelperFunctions:
    """Test helper functions."""

    def test_parse_date_mm_dd_yyyy(self):
        """Test parsing MM/DD/YYYY format."""
        result = parse_date('01/31/2024')
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 31

    def test_parse_date_m_d_yyyy(self):
        """Test parsing M/D/YYYY format."""
        result = parse_date('1/31/2024')
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024

    def test_parse_date_yyyy_mm_dd(self):
        """Test parsing YYYY-MM-DD format."""
        result = parse_date('2024-01-31')
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024

    def test_parse_date_none(self):
        """Test parsing None returns NaT."""
        result = parse_date(None)
        assert pd.isna(result)

    def test_end_of_month(self):
        """Test end of month calculation."""
        date = pd.Timestamp('2024-01-15')
        result = end_of_month(date)
        assert result.day == 31
        assert result.month == 1

    def test_end_of_month_february_leap_year(self):
        """Test end of month for February in leap year."""
        date = pd.Timestamp('2024-02-01')
        result = end_of_month(date)
        assert result.day == 29
        assert result.month == 2

    def test_clean_cohort_int(self):
        """Test cleaning integer cohort."""
        result = clean_cohort(202001)
        assert result == '202001'

    def test_clean_cohort_float(self):
        """Test cleaning float cohort."""
        result = clean_cohort(202001.0)
        assert result == '202001'

    def test_clean_cohort_string(self):
        """Test cleaning string cohort."""
        result = clean_cohort('202001')
        assert result == '202001'

    def test_clean_cohort_none(self):
        """Test cleaning None cohort."""
        result = clean_cohort(None)
        assert result == ''

    def test_safe_divide_normal(self):
        """Test normal division."""
        result = safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_zero_denominator(self):
        """Test division by zero."""
        result = safe_divide(10, 0)
        assert result == 0.0

    def test_safe_divide_none_values(self):
        """Test division with None values."""
        result = safe_divide(None, 2)
        assert result == 0.0
        result = safe_divide(10, None)
        assert result == 0.0


# =============================================================================
# UNIT TESTS: DATA LOADING
# =============================================================================

class TestDataLoading:
    """Test data loading functions."""

    def test_load_fact_raw(self, fact_raw_path):
        """Test loading fact raw data."""
        df = load_fact_raw(fact_raw_path)

        assert len(df) > 0
        assert 'CalendarMonth' in df.columns
        assert 'Cohort' in df.columns
        assert 'Segment' in df.columns
        assert 'MOB' in df.columns
        assert 'OpeningGBV' in df.columns

        # Check data types
        assert df['CalendarMonth'].dtype == 'datetime64[ns]'
        assert df['Cohort'].dtype == 'object'
        assert df['MOB'].dtype in ['int64', 'int32']
        assert df['OpeningGBV'].dtype == 'float64'

    def test_load_fact_raw_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_fact_raw('non_existent_file.csv')

    def test_load_rate_methodology(self, methodology_path):
        """Test loading rate methodology."""
        df = load_rate_methodology(methodology_path)

        assert len(df) > 0
        assert 'Segment' in df.columns
        assert 'Cohort' in df.columns
        assert 'Metric' in df.columns
        assert 'Approach' in df.columns
        assert 'MOB_Start' in df.columns
        assert 'MOB_End' in df.columns

    def test_load_debt_sale_schedule(self, debt_sale_path):
        """Test loading debt sale schedule."""
        df = load_debt_sale_schedule(debt_sale_path)

        assert df is not None
        assert len(df) > 0
        assert 'ForecastMonth' in df.columns
        assert 'Debt_Sale_WriteOffs' in df.columns

    def test_load_debt_sale_schedule_none(self):
        """Test loading None path returns None."""
        result = load_debt_sale_schedule(None)
        assert result is None


# =============================================================================
# UNIT TESTS: CURVES CALCULATION
# =============================================================================

class TestCurvesCalculation:
    """Test curves calculation functions."""

    def test_calculate_curves_base(self, fact_raw_df):
        """Test calculating base curves."""
        curves = calculate_curves_base(fact_raw_df)

        assert len(curves) > 0
        assert 'Segment' in curves.columns
        assert 'Cohort' in curves.columns
        assert 'MOB' in curves.columns
        assert 'Coll_Principal_Rate' in curves.columns
        assert 'InterestRevenue_Rate' in curves.columns

    def test_curves_rates_reasonable(self, fact_raw_df):
        """Test that calculated rates are reasonable."""
        curves = calculate_curves_base(fact_raw_df)

        # Collection rates should be negative (money leaving)
        assert curves['Coll_Principal_Rate'].max() <= 0

        # Interest revenue rate should be positive
        assert curves['InterestRevenue_Rate'].min() >= 0

    def test_extend_curves(self, curves_base_df):
        """Test extending curves."""
        extended = extend_curves(curves_base_df, 12)

        # Should have more rows than base
        assert len(extended) > len(curves_base_df)

        # Should have extended MOBs
        max_base_mob = curves_base_df['MOB'].max()
        max_extended_mob = extended['MOB'].max()
        assert max_extended_mob > max_base_mob


# =============================================================================
# UNIT TESTS: SEED GENERATION
# =============================================================================

class TestSeedGeneration:
    """Test seed generation functions."""

    def test_generate_seed_curves(self, fact_raw_df):
        """Test generating seed curves."""
        seed = generate_seed_curves(fact_raw_df)

        assert len(seed) > 0
        assert 'Segment' in seed.columns
        assert 'Cohort' in seed.columns
        assert 'MOB' in seed.columns
        assert 'BoM' in seed.columns
        assert 'ForecastMonth' in seed.columns

        # All BoM should be positive (filtered out zeros)
        assert (seed['BoM'] > 0).all()

    def test_generate_impairment_seed(self, fact_raw_df):
        """Test generating impairment seed."""
        seed = generate_impairment_seed(fact_raw_df)

        assert len(seed) > 0
        assert 'Segment' in seed.columns
        assert 'Cohort' in seed.columns
        assert 'Prior_Provision_Balance' in seed.columns


# =============================================================================
# UNIT TESTS: METHODOLOGY LOOKUP
# =============================================================================

class TestMethodologyLookup:
    """Test methodology lookup functions."""

    def test_get_specificity_score_exact_match(self):
        """Test specificity score for exact matches."""
        row = pd.Series({
            'Segment': 'NRP-S',
            'Cohort': '202001',
            'Metric': 'Coll_Principal',
            'MOB_Start': 0,
            'MOB_End': 999,
        })

        score = get_specificity_score(row, 'NRP-S', '202001', 'Coll_Principal', 50)

        # Should get points for segment (8) + cohort (4) + metric (2) + mob range
        assert score > 14

    def test_get_specificity_score_all_wildcard(self):
        """Test specificity score for ALL wildcards."""
        row = pd.Series({
            'Segment': 'ALL',
            'Cohort': 'ALL',
            'Metric': 'Coll_Principal',
            'MOB_Start': 0,
            'MOB_End': 999,
        })

        score = get_specificity_score(row, 'NRP-S', '202001', 'Coll_Principal', 50)

        # Should only get points for metric (2) + mob range
        assert score < 3

    def test_get_methodology(self, methodology_df):
        """Test getting methodology for a cohort."""
        result = get_methodology(
            methodology_df,
            segment='NRP-S',
            cohort='202001',
            mob=50,
            metric='Coll_Principal'
        )

        assert 'Approach' in result
        assert result['Approach'] in Config.VALID_APPROACHES + ['NoMatch_ERROR']


# =============================================================================
# UNIT TESTS: RATE CALCULATION
# =============================================================================

class TestRateCalculation:
    """Test rate calculation functions."""

    def test_fn_cohort_avg(self, curves_base_df):
        """Test cohort average calculation."""
        rate = fn_cohort_avg(
            curves_base_df,
            segment='NRP-S',
            cohort='202001',
            mob=55,
            metric_col='Coll_Principal_Rate',
            lookback=6
        )

        # Should return a rate or None
        if rate is not None:
            assert isinstance(rate, float)
            assert -1 <= rate <= 0  # Collections are negative

    def test_fn_cohort_trend(self, curves_base_df):
        """Test cohort trend calculation."""
        rate = fn_cohort_trend(
            curves_base_df,
            segment='NRP-S',
            cohort='202001',
            mob=60,
            metric_col='Coll_Principal_Rate'
        )

        # Should return a rate or None
        if rate is not None:
            assert isinstance(rate, float)

    def test_fn_seg_median(self, curves_base_df):
        """Test segment median calculation."""
        rate = fn_seg_median(
            curves_base_df,
            segment='NRP-S',
            mob=50,
            metric_col='Coll_Principal_Rate'
        )

        # Should return a rate or None
        if rate is not None:
            assert isinstance(rate, float)

    def test_apply_rate_cap(self):
        """Test rate cap application."""
        # Test capping above max
        rate = apply_rate_cap(-0.20, 'Coll_Principal', 'CohortAvg')
        assert rate == -0.15  # Max is -0.15

        # Test capping below min
        rate = apply_rate_cap(0.01, 'Coll_Principal', 'CohortAvg')
        assert rate == 0.0  # Min is 0

        # Test Manual bypass
        rate = apply_rate_cap(-0.20, 'Coll_Principal', 'Manual')
        assert rate == -0.20  # Manual bypasses caps


# =============================================================================
# UNIT TESTS: OUTPUT GENERATION
# =============================================================================

class TestOutputGeneration:
    """Test output generation functions."""

    def test_generate_validation_output_empty(self):
        """Test validation output with empty DataFrame."""
        recon, val = generate_validation_output(pd.DataFrame())

        assert len(recon) == 0
        assert len(val) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflow."""

    def test_full_forecast_short(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test full forecast workflow with 3 months."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=3
        )

        assert len(forecast) > 0
        assert 'ForecastMonth' in forecast.columns
        assert 'ClosingGBV' in forecast.columns
        assert 'ClosingNBV' in forecast.columns

        # Check output files exist
        assert os.path.exists(os.path.join(temp_output_dir, 'Forecast_Summary.xlsx'))
        assert os.path.exists(os.path.join(temp_output_dir, 'Forecast_Details.xlsx'))
        assert os.path.exists(os.path.join(temp_output_dir, 'Impairment_Analysis.xlsx'))
        assert os.path.exists(os.path.join(temp_output_dir, 'Validation_Report.xlsx'))

    def test_gbv_reconciliation(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test GBV reconciliation passes."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=3
        )

        # Calculate expected closing GBV
        expected_gbv = (
            forecast['OpeningGBV'] +
            forecast['InterestRevenue'] -
            abs(forecast['Coll_Principal']) -
            abs(forecast['Coll_Interest']) -
            forecast['WO_DebtSold'] -
            forecast['WO_Other']
        )

        # Variance should be minimal
        variance = (expected_gbv - forecast['ClosingGBV']).abs().max()
        assert variance < 0.01

    def test_nbv_reconciliation(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test NBV reconciliation passes."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=3
        )

        # Calculate expected closing NBV
        expected_nbv = forecast['ClosingGBV'] - forecast['Net_Impairment']

        # Variance should be minimal
        variance = (expected_nbv - forecast['ClosingNBV']).abs().max()
        assert variance < 0.01

    def test_no_nan_values(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test no NaN values in key columns."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=3
        )

        key_columns = ['OpeningGBV', 'ClosingGBV', 'ClosingNBV', 'Net_Impairment']
        for col in key_columns:
            assert not forecast[col].isna().any(), f"Found NaN values in {col}"

    def test_forecast_chain_continuity(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test forecast chain continuity (ClosingGBV[t] = OpeningGBV[t+1])."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=3
        )

        # Group by Segment and Cohort
        for (segment, cohort), group in forecast.groupby(['Segment', 'Cohort']):
            group = group.sort_values('ForecastMonth').reset_index(drop=True)

            for i in range(len(group) - 1):
                closing_gbv = group.loc[i, 'ClosingGBV']
                next_opening_gbv = group.loc[i + 1, 'OpeningGBV']

                variance = abs(closing_gbv - next_opening_gbv)
                assert variance < 0.01, f"Chain break for {segment}/{cohort}: {closing_gbv} != {next_opening_gbv}"

    def test_with_debt_sale_schedule(self, fact_raw_path, methodology_path,
                                     debt_sale_path, temp_output_dir):
        """Test forecast with debt sale schedule."""
        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=debt_sale_path,
            output_dir=temp_output_dir,
            max_months=12
        )

        assert len(forecast) > 0
        assert 'Debt_Sale_WriteOffs' in forecast.columns
        assert 'Debt_Sale_Impact' in forecast.columns


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests."""

    def test_forecast_performance(self, fact_raw_path, methodology_path, temp_output_dir):
        """Test that forecast completes within performance target."""
        import time

        start = time.time()

        forecast = run_backbook_forecast(
            fact_raw_path=fact_raw_path,
            methodology_path=methodology_path,
            debt_sale_path=None,
            output_dir=temp_output_dir,
            max_months=12
        )

        elapsed = time.time() - start

        # Should complete within 30 seconds for sample data
        assert elapsed < 30, f"Forecast took {elapsed:.2f} seconds, expected < 30"

        print(f"Forecast completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
