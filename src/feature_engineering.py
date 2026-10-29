"""
src/feature_engineering.py
───────────────────────────
Build time-series features for regression and classification.

All lag/rolling features use shift() so only past information is used.
The chronological split is strict — no random shuffling.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Time features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features derived from the date column."""
    df = df.copy()
    dt = df[date_col]
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = dt.dt.month                     # 1–12
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = dt.dt.day
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Lag features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "revenue",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged values of *target_col*.

    lag_N  =  value N days ago  (uses .shift(N) — no future data).
    NaN rows introduced at the head are dropped later during split.
    """
    if lags is None:
        lags = config.LAG_DAYS
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling features
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "revenue",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean features.

    rolling_mean_N  =  mean of the N days *before* the current day.
    Uses closed='left' (i.e., shift(1).rolling(N)) to avoid leakage.
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS
    df = df.copy()
    for w in windows:
        # shift(1) ensures we don't use the current day's value
        df[f"rolling_mean_{w}"] = (
            df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full feature pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to daily aggregated data.

    Steps (in order):
      1. Time features
      2. Lag features
      3. Rolling mean features
      4. Drop rows with NaN (introduced by lags)
    """
    df = add_time_features(daily_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    logger.info("Feature engineering: dropped %d NaN rows → %d usable rows.",
                before - after, after)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Chronological train / validation / test split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / validation / test by position (time order).

    NEVER uses random shuffling for time-series data.

    Returns
    -------
    train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split → Train: %d rows (%s–%s)  "
        "| Val: %d rows (%s–%s)  "
        "| Test: %d rows (%s–%s)",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(val),   val["date"].min().date(),   val["date"].max().date(),
        len(test),  test["date"].min().date(),  test["date"].max().date(),
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns list
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (excluding date and target)."""
    exclude = {"date", "revenue"}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# Classification target
# ─────────────────────────────────────────────────────────────────────────────

def add_classification_target(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str = "revenue",
) -> Tuple[pd.DataFrame, float]:
    """
    Add binary 'high_sales' column.

    Threshold = median revenue of the *training set only* (no leakage).
    high_sales = 1 if revenue >= threshold, else 0.

    Returns (df_with_target, threshold).
    """
    threshold = float(train_df[target_col].median())
    df = df.copy()
    df["high_sales"] = (df[target_col] >= threshold).astype(int)
    logger.info(
        "Classification target: threshold=£%.2f  "
        "(high=1: %d  low=0: %d)",
        threshold,
        (df["high_sales"] == 1).sum(),
        (df["high_sales"] == 0).sum(),
    )
    return df, threshold
