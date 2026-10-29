"""
src/forecasting.py
──────────────────
Multi-step ahead revenue forecasting.

Strategy: recursive / iterated one-step-ahead forecasting.
  For each future day t:
    1. Build features using all known/previously-predicted values
    2. Predict revenue for day t
    3. Append predicted revenue to the history for the next step

This correctly avoids using actual future sales to create forecast inputs.
"""
import logging
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.feature_engineering import add_time_features

logger = logging.getLogger(__name__)


def _load_best_model():
    """Load Random Forest (best regressor) if available, else Linear Regression."""
    if os.path.exists(config.RF_REGRESSOR_PATH):
        return joblib.load(config.RF_REGRESSOR_PATH), "Random Forest Regressor", False
    if os.path.exists(config.LINEAR_MODEL_PATH):
        model = joblib.load(config.LINEAR_MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH) if os.path.exists(config.SCALER_PATH) else None
        return model, "Linear Regression", scaler
    raise FileNotFoundError("No trained model found. Run training first.")


def generate_forecast(
    daily_df: pd.DataFrame,
    horizon_days: int = config.FORECAST_HORIZON_DEFAULT,
    model_name: str | None = None,
) -> pd.DataFrame:
    """
    Generate a *horizon_days*-ahead revenue forecast using recursive prediction.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Full daily aggregated historical data (date, revenue, …).
    horizon_days : int
        Number of future days to forecast.
    model_name : str | None
        'Random Forest Regressor' or 'Linear Regression'.
        If None, uses best available model.

    Returns
    -------
    pd.DataFrame
        Columns: date, predicted_revenue, model_name, is_forecast (True).
    """
    model, name, scaler = _load_best_model()
    if model_name and model_name != name:
        logger.warning("Requested model '%s' not available; using '%s'.", model_name, name)

    # Working history — we'll extend this for each step
    history = daily_df[["date", "revenue"]].copy().sort_values("date").reset_index(drop=True)

    last_date = history["date"].max()
    forecast_rows = []

    for step in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=step)

        # Build a one-row DataFrame to hold features for this future date
        row = pd.DataFrame({"date": [future_date], "revenue": [np.nan]})

        # Add static features (quantity, num_transactions, unique_products) from the most recent known record
        static_cols = [c for c in daily_df.columns if c not in {"date", "revenue"}]
        latest_record = daily_df.iloc[-1]
        for col in static_cols:
            row[col] = latest_record[col]

        # Time features
        row = add_time_features(row)

        # Lag features using history (past revenue values only)
        for lag in config.LAG_DAYS:
            idx = len(history) - lag
            row[f"lag_{lag}"] = history["revenue"].iloc[idx] if idx >= 0 else np.nan

        # Rolling mean features
        for w in config.ROLLING_WINDOWS:
            vals = history["revenue"].iloc[-w:].values
            row[f"rolling_mean_{w}"] = vals.mean() if len(vals) > 0 else np.nan
        # Feature columns (same order as training)
        feature_cols = [c for c in row.columns if c not in {"date", "revenue"}]
        X = row[feature_cols].fillna(0).values

        if scaler:
            X = scaler.transform(X)

        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # revenue cannot be negative

        forecast_rows.append(
            {
                "date": future_date,
                "predicted_revenue": round(pred, 2),
                "model_name": name,
                "is_forecast": True,
                "horizon_days": horizon_days,
            }
        )

        # Append predicted value to history for the next step
        history = pd.concat(
            [history, pd.DataFrame({"date": [future_date], "revenue": [pred]})],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(forecast_rows)
    logger.info(
        "Generated %d-day forecast using '%s'. Predicted range: £%.0f – £%.0f",
        horizon_days, name,
        forecast_df["predicted_revenue"].min(),
        forecast_df["predicted_revenue"].max(),
    )
    return forecast_df


def save_forecast_csv(forecast_df: pd.DataFrame, filepath: str) -> None:
    forecast_df["date"] = forecast_df["date"].astype(str)
    forecast_df.to_csv(filepath, index=False)
    logger.info("Forecast saved → %s", filepath)
