"""
src/data_loader.py
──────────────────
Load the raw UCI Online Retail dataset from disk.
Handles both .xlsx and the alternative .csv format.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def _load_xlsx_low_memory(filepath: str) -> pd.DataFrame:
    """
    Stream .xlsx to a temp CSV on disk, then read CSV with pandas.

    Writing row-by-row to disk avoids ever holding all 541k rows in RAM,
    which is critical when Windows virtual memory is constrained (C: full).
    """
    import csv
    import openpyxl

    logger.info("Using low-memory xlsx->csv streaming reader …")

    # Write temp CSV on D: drive (same folder as the source file)
    tmp_path = os.path.join(os.path.dirname(filepath), "_online_retail_tmp.csv")

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            writer.writerow(row)

    wb.close()
    logger.info("Streamed Excel to temp CSV: %s", tmp_path)

    df = pd.read_csv(
        tmp_path,
        encoding="utf-8",
        dtype={
            "InvoiceNo":   str,
            "StockCode":   str,
            "Description": str,
            "CustomerID":  str,
            "Country":     str,
        },
    )

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    logger.info("Loaded %d rows from temp CSV.", len(df))
    return df


def load_raw_data(filepath: str) -> pd.DataFrame:

    """
    Load the raw Online Retail dataset.

    Parameters
    ----------
    filepath : str
        Absolute path to ``Online Retail.xlsx`` (or .csv variant).

    Returns
    -------
    pd.DataFrame
        Raw data with InvoiceDate parsed to datetime.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Download 'Online Retail.xlsx' from "
            "https://archive.ics.uci.edu/dataset/352/online+retail "
            "and place it in the data/ directory."
        )

    ext = os.path.splitext(filepath)[1].lower()
    logger.info("Loading dataset from %s …", filepath)

    if ext in (".xlsx", ".xls"):
        df = _load_xlsx_low_memory(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath, encoding="ISO-8859-1", dtype={"CustomerID": str})
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Validate columns
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Parse InvoiceDate — cache=False avoids allocating a cache array
    # (which fails when virtual memory is tight due to full C: drive)
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce", cache=False
    )
    df = df.dropna(subset=["InvoiceDate"])

    logger.info("Raw dataset loaded: %d rows, %d columns.", *df.shape)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for logging / reporting."""
    return {
        "total_rows": len(df),
        "date_min": str(df["InvoiceDate"].min().date()),
        "date_max": str(df["InvoiceDate"].max().date()),
        "unique_invoices": df["InvoiceNo"].nunique(),
        "unique_products": df["StockCode"].nunique(),
        "unique_customers": df["CustomerID"].nunique(),
        "unique_countries": df["Country"].nunique(),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_description": int(df["Description"].isna().sum()),
        "negative_quantity": int((df["Quantity"] < 0).sum()),
        "zero_or_neg_price": int((df["UnitPrice"] <= 0).sum()),
    }
