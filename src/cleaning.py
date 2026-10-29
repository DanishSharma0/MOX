"""
src/cleaning.py
───────────────
Clean raw UCI Online Retail transactions and aggregate to daily revenue.

Cleaning rules (documented):
  1. Drop rows where InvoiceNo starts with 'C'  → cancellations
  2. Drop rows where Quantity <= 0              → returns / data errors
  3. Drop rows where UnitPrice <= 0             → invalid prices
  4. Drop rows with missing Description         → unidentifiable product
  5. Add Revenue = Quantity × UnitPrice
  6. Aggregate by InvoiceDate (date only) → daily totals
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Transaction-level cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning rules to the raw transaction DataFrame.

    Returns a cleaned copy.  Does NOT modify df in-place.
    """
    original_count = len(df)
    df = df.copy()

    # Rule 1: drop cancellations (InvoiceNo starts with 'C')
    cancellations_mask = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~cancellations_mask]
    logger.info("After removing cancellations: %d rows (removed %d).",
                len(df), cancellations_mask.sum())

    # Rule 2: drop non-positive quantities
    qty_mask = df["Quantity"] <= 0
    df = df[~qty_mask]
    logger.info("After removing non-positive quantities: %d rows (removed %d).",
                len(df), qty_mask.sum())

    # Rule 3: drop non-positive prices
    price_mask = df["UnitPrice"] <= 0
    df = df[~price_mask]
    logger.info("After removing non-positive prices: %d rows (removed %d).",
                len(df), price_mask.sum())

    # Rule 4: drop missing descriptions
    desc_mask = df["Description"].isna() | (df["Description"].str.strip() == "")
    df = df[~desc_mask]
    logger.info("After removing missing descriptions: %d rows (removed %d).",
                len(df), desc_mask.sum())

    # Rule 5: revenue
    df = calculate_revenue(df)

    logger.info(
        "Cleaning complete.  %d → %d rows (kept %.1f%%).",
        original_count, len(df), 100 * len(df) / original_count,
    )
    return df.reset_index(drop=True)


def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add Revenue column = Quantity × UnitPrice."""
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Daily aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned transaction-level data to daily totals.

    Returns a DataFrame indexed from 0 with columns:
        date, revenue, quantity, num_transactions, unique_products
    Sorted chronologically.
    """
    df = df.copy()
    df["date"] = df["InvoiceDate"].dt.date  # date only, no time

    daily = (
        df.groupby("date")
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            num_transactions=("InvoiceNo", "nunique"),
            unique_products=("StockCode", "nunique"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    logger.info(
        "Daily aggregation: %d days  |  %s → %s  |  total revenue £%.2f",
        len(daily),
        daily["date"].min().date(),
        daily["date"].max().date(),
        daily["revenue"].sum(),
    )
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning summary (for reports / UI)
# ─────────────────────────────────────────────────────────────────────────────

def cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Return a summary dict comparing raw vs. cleaned counts."""
    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "removed_rows": len(raw_df) - len(clean_df),
        "retention_pct": round(100 * len(clean_df) / len(raw_df), 2),
        "cancellations_removed": int(
            raw_df["InvoiceNo"].astype(str).str.startswith("C").sum()
        ),
        "negative_qty_removed": int((raw_df["Quantity"] <= 0).sum()),
        "invalid_price_removed": int((raw_df["UnitPrice"] <= 0).sum()),
        "missing_desc_removed": int(
            (raw_df["Description"].isna()
             | (raw_df["Description"].astype(str).str.strip() == "")).sum()
        ),
    }
