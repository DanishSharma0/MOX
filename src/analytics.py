"""
src/analytics.py
────────────────
Sales analytics queries for the dashboard.
Top products, monthly trends, revenue distributions, etc.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def top_products_from_df(
    clean_df: pd.DataFrame,
    n: int = 10,
    by: str = "Revenue",
) -> pd.DataFrame:
    """
    Return top N products by total revenue (or quantity) from the
    cleaned transaction-level DataFrame.
    """
    grouped = (
        clean_df.groupby(["StockCode", "Description"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_quantity=("Quantity", "sum"),
            transaction_count=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(n)
    )
    return grouped


def monthly_revenue_from_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily revenue DataFrame to monthly summary.
    """
    df = daily_df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("year_month")
        .agg(
            monthly_revenue=("revenue", "sum"),
            monthly_quantity=("quantity", "sum"),
            active_days=("date", "count"),
        )
        .reset_index()
    )
    return monthly


def revenue_distribution_stats(daily_df: pd.DataFrame) -> dict:
    """Return statistical summary of daily revenue."""
    s = daily_df["revenue"]
    return {
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "q25": round(float(s.quantile(0.25)), 2),
        "q75": round(float(s.quantile(0.75)), 2),
    }
