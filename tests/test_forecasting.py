"""
tests/test_forecasting.py — Tests for the forecasting module.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

# We need a trained model for forecasting tests.
# Skip if models aren't trained yet.
import config


def make_daily_df(n=100):
    dates = pd.date_range("2010-12-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": dates,
        "revenue": rng.uniform(5000, 25000, n),
        "quantity": rng.integers(100, 1000, n),
        "num_transactions": rng.integers(10, 100, n),
        "unique_products": rng.integers(50, 300, n),
    })


def test_forecast_requires_trained_model():
    """Should raise FileNotFoundError if no model is available."""
    # Temporarily rename models dir if it exists
    from src.forecasting import generate_forecast
    daily = make_daily_df()
    if not (
        os.path.exists(config.RF_REGRESSOR_PATH)
        or os.path.exists(config.LINEAR_MODEL_PATH)
    ):
        with pytest.raises(FileNotFoundError):
            generate_forecast(daily, horizon_days=7)


@pytest.mark.skipif(
    not (
        os.path.exists(config.RF_REGRESSOR_PATH)
        or os.path.exists(config.LINEAR_MODEL_PATH)
    ),
    reason="No trained model available",
)
def test_forecast_output_shape():
    from src.forecasting import generate_forecast
    daily = make_daily_df()
    horizon = 14
    fc = generate_forecast(daily, horizon_days=horizon)
    assert len(fc) == horizon


@pytest.mark.skipif(
    not (
        os.path.exists(config.RF_REGRESSOR_PATH)
        or os.path.exists(config.LINEAR_MODEL_PATH)
    ),
    reason="No trained model available",
)
def test_forecast_dates_are_future():
    from src.forecasting import generate_forecast
    daily = make_daily_df()
    fc = generate_forecast(daily, horizon_days=7)
    last_hist_date = daily["date"].max()
    assert (pd.to_datetime(fc["date"]) > last_hist_date).all()


@pytest.mark.skipif(
    not (
        os.path.exists(config.RF_REGRESSOR_PATH)
        or os.path.exists(config.LINEAR_MODEL_PATH)
    ),
    reason="No trained model available",
)
def test_forecast_revenue_non_negative():
    from src.forecasting import generate_forecast
    daily = make_daily_df()
    fc = generate_forecast(daily, horizon_days=7)
    assert (fc["predicted_revenue"] >= 0).all()


@pytest.mark.skipif(
    not (
        os.path.exists(config.RF_REGRESSOR_PATH)
        or os.path.exists(config.LINEAR_MODEL_PATH)
    ),
    reason="No trained model available",
)
def test_forecast_columns():
    from src.forecasting import generate_forecast
    daily = make_daily_df()
    fc = generate_forecast(daily, horizon_days=7)
    for col in ["date", "predicted_revenue", "model_name"]:
        assert col in fc.columns
