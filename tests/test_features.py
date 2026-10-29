"""
tests/test_features.py — Tests for feature engineering module.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    chronological_split,
    get_feature_columns,
    add_classification_target,
)


def make_daily_df(n=60):
    """Create a simple daily revenue DataFrame with n days."""
    dates = pd.date_range("2010-12-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    revenue = rng.uniform(5000, 25000, size=n)
    return pd.DataFrame({
        "date": dates,
        "revenue": revenue,
        "quantity": rng.integers(200, 1000, size=n),
        "num_transactions": rng.integers(10, 100, size=n),
        "unique_products": rng.integers(50, 300, size=n),
    })


def test_add_time_features_columns():
    df = make_daily_df()
    result = add_time_features(df)
    for col in ["day_of_week", "month", "week_of_year", "quarter", "is_weekend", "day_of_month"]:
        assert col in result.columns, f"Missing: {col}"


def test_day_of_week_range():
    df = make_daily_df()
    result = add_time_features(df)
    assert result["day_of_week"].between(0, 6).all()


def test_month_range():
    df = make_daily_df()
    result = add_time_features(df)
    assert result["month"].between(1, 12).all()


def test_is_weekend_binary():
    df = make_daily_df()
    result = add_time_features(df)
    assert set(result["is_weekend"].unique()).issubset({0, 1})


def test_add_lag_features_columns():
    df = make_daily_df()
    result = add_lag_features(df, lags=[1, 7, 14])
    for lag in [1, 7, 14]:
        assert f"lag_{lag}" in result.columns


def test_lag_values_are_shifted():
    """lag_1[i] should equal revenue[i-1]."""
    df = make_daily_df(30)
    result = add_lag_features(df, lags=[1])
    # Check from index 1 onwards (index 0 is NaN)
    for i in range(1, len(result)):
        assert result["lag_1"].iloc[i] == pytest.approx(result["revenue"].iloc[i - 1])


def test_rolling_features_no_future_leakage():
    """rolling_mean_7 for day i should only use days <= i-1."""
    df = make_daily_df(30)
    result = add_rolling_features(df, windows=[7])
    # Manually check: rolling mean at index 7 should be mean of indices 0..6
    expected = df["revenue"].iloc[0:7].mean()
    actual = result["rolling_mean_7"].iloc[7]
    assert actual == pytest.approx(expected, rel=1e-3)


def test_chronological_split_order():
    """Ensure train comes before val comes before test in time."""
    df = make_daily_df(60)
    feat = build_features(df)
    train, val, test = chronological_split(feat, train_ratio=0.7, val_ratio=0.15)
    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()


def test_chronological_split_no_overlap():
    df = make_daily_df(60)
    feat = build_features(df)
    train, val, test = chronological_split(feat)
    train_dates = set(train["date"].dt.date)
    val_dates   = set(val["date"].dt.date)
    test_dates  = set(test["date"].dt.date)
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


def test_get_feature_columns_excludes_target():
    df = make_daily_df(60)
    feat = build_features(df)
    cols = get_feature_columns(feat)
    assert "revenue" not in cols
    assert "date" not in cols


def test_classification_target_uses_train_threshold():
    """Threshold must come from train_df only."""
    df = make_daily_df(60)
    feat = build_features(df)
    train, val, test = chronological_split(feat)
    train_cls, threshold = add_classification_target(train, train)
    test_cls, t2 = add_classification_target(test, train)
    # Same threshold applied
    assert threshold == t2
    # Threshold equals training median
    assert threshold == pytest.approx(train["revenue"].median(), rel=1e-6)


def test_no_nan_after_build_features():
    df = make_daily_df(60)
    feat = build_features(df)
    assert feat.isnull().sum().sum() == 0
