"""
src/train_regression.py
───────────────────────
Train and evaluate regression models on the daily revenue time-series.

Models:
  1. Naive baseline — previous day's revenue
  2. Linear Regression
  3. Random Forest Regressor

All models use a chronological train / validation / test split.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.evaluation import evaluate_regression

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].values
    y = df["revenue"].values
    return X, y


def save_model(model, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved → %s", filepath)


def load_model(filepath: str):
    return joblib.load(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────────

def baseline_previous_day(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Naive baseline: predict today's revenue = yesterday's actual revenue.
    Uses lag_1 feature (which represents the previous day's revenue).
    """
    if "lag_1" not in test_df.columns:
        raise ValueError("lag_1 feature is required for the baseline model.")
    y_pred = test_df["lag_1"].values
    y_true = test_df["revenue"].values
    metrics = evaluate_regression(y_true, y_pred)
    metrics["model"] = "Baseline (Lag-1)"
    logger.info("Baseline → R²=%.4f  MAE=£%.2f", metrics["r2"], metrics["mae"])
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[LinearRegression, StandardScaler, dict]:
    """
    Train Linear Regression with StandardScaler (fitted on train only).
    Returns (model, scaler, metrics_dict).
    """
    X_train, y_train = _xy(train_df, feature_cols)
    X_test, y_test = _xy(test_df, feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = evaluate_regression(y_test, y_pred)
    metrics["model"] = "Linear Regression"
    logger.info("Linear Regression → R²=%.4f  MAE=£%.2f", metrics["r2"], metrics["mae"])

    # Save artefacts
    save_model(model, config.LINEAR_MODEL_PATH)
    save_model(scaler, config.SCALER_PATH)

    return model, scaler, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Regressor
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest_regressor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict]:
    """
    Train Random Forest Regressor (no scaling needed).
    Returns (model, metrics_dict).
    """
    X_train, y_train = _xy(train_df, feature_cols)
    X_test, y_test = _xy(test_df, feature_cols)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    metrics["model"] = "Random Forest Regressor"
    logger.info("Random Forest → R²=%.4f  MAE=£%.2f", metrics["r2"], metrics["mae"])

    save_model(model, config.RF_REGRESSOR_PATH)
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Run all regression models
# ─────────────────────────────────────────────────────────────────────────────

def run_all_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    """
    Train all three regression models and return list of metric dicts.
    Uses val_df + test_df combined as held-out evaluation set for final metrics.
    """
    # Combine val and test for final evaluation (train on train only)
    eval_df = pd.concat([val_df, test_df], ignore_index=True)

    metrics_list = []

    # 1. Baseline
    baseline_m = baseline_previous_day(train_df, eval_df, feature_cols)
    metrics_list.append(baseline_m)

    # 2. Linear Regression
    _, _, lr_m = train_linear_regression(train_df, eval_df, feature_cols)
    metrics_list.append(lr_m)

    # 3. Random Forest
    _, rf_m = train_random_forest_regressor(train_df, eval_df, feature_cols)
    metrics_list.append(rf_m)

    return metrics_list
