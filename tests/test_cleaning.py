"""
tests/test_cleaning.py — Tests for data cleaning module.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from src.cleaning import (
    calculate_revenue,
    clean_transactions,
    aggregate_daily_revenue,
    cleaning_summary,
)


def make_sample_df():
    """Create a small transaction DataFrame for testing."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536379", "536366", "536367", "536368", "536369"],
        "StockCode": ["85123A", "85123A", "71053", "84406B", "84029G", "84029E"],
        "Description": ["WHITE HANGING HEART", "CANCEL", "WHITE METAL LANTERN",
                         "CREAM CUPID HEARTS COAT HANGER", "KNITTED UNION FLAG HOT WATER BOTTLE",
                         ""],
        "Quantity": [6, -6, 8, 6, 6, 2],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01", "2010-12-01", "2010-12-01",
            "2010-12-01", "2010-12-01", "2010-12-01",
        ]),
        "UnitPrice": [2.55, 2.55, 3.39, 2.75, 3.39, 0.0],
        "CustomerID": ["17850", "17850", "17850", "13047", "13047", "13047"],
        "Country": ["United Kingdom"] * 6,
    })


def test_calculate_revenue():
    df = make_sample_df()
    result = calculate_revenue(df)
    assert "Revenue" in result.columns
    # Row 0: 6 * 2.55 = 15.30
    assert abs(result.loc[0, "Revenue"] - 15.30) < 0.01


def test_clean_removes_cancellations():
    df = make_sample_df()
    clean = clean_transactions(df)
    assert not any(clean["InvoiceNo"].astype(str).str.startswith("C"))


def test_clean_removes_negative_quantity():
    df = make_sample_df()
    clean = clean_transactions(df)
    assert (clean["Quantity"] > 0).all()


def test_clean_removes_zero_price():
    df = make_sample_df()
    clean = clean_transactions(df)
    assert (clean["UnitPrice"] > 0).all()


def test_clean_removes_missing_description():
    df = make_sample_df()
    clean = clean_transactions(df)
    assert clean["Description"].notna().all()
    assert not (clean["Description"].str.strip() == "").any()


def test_clean_revenue_positive():
    df = make_sample_df()
    clean = clean_transactions(df)
    assert (clean["Revenue"] > 0).all()


def test_aggregate_daily_revenue_shape():
    df = make_sample_df()
    clean = clean_transactions(df)
    daily = aggregate_daily_revenue(clean)
    assert "date" in daily.columns
    assert "revenue" in daily.columns
    assert len(daily) >= 1


def test_aggregate_revenue_sum():
    """Revenue in daily should equal sum of individual revenues."""
    df = make_sample_df()
    clean = clean_transactions(df)
    daily = aggregate_daily_revenue(clean)
    assert abs(daily["revenue"].sum() - clean["Revenue"].sum()) < 0.01


def test_cleaning_summary_keys():
    df = make_sample_df()
    clean = clean_transactions(df)
    summary = cleaning_summary(df, clean)
    assert "raw_rows" in summary
    assert "clean_rows" in summary
    assert "retention_pct" in summary


def test_no_modification_of_original():
    """clean_transactions should not modify the original DataFrame."""
    df = make_sample_df()
    original_len = len(df)
    _ = clean_transactions(df)
    assert len(df) == original_len
