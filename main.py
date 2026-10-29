"""
main.py
───────
Full end-to-end pipeline:
  1. Load & clean raw data
  2. Import to MongoDB
  3. Feature engineering
  4. Chronological split
  5. Train regression models
  6. Train classification models
  7. Save metrics + charts
  8. Generate forecast
  9. Generate PDF report

Run:  python main.py
"""
import json
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from src.data_loader import load_raw_data, dataset_summary
from src.cleaning import clean_transactions, aggregate_daily_revenue, cleaning_summary
from src.feature_engineering import (
    build_features,
    chronological_split,
    get_feature_columns,
    add_classification_target,
)
from src.train_regression import run_all_regression
from src.train_classification import run_all_classification
from src.evaluation import (
    save_regression_metrics_csv,
    save_classification_metrics_csv,
    plot_model_comparison,
    plot_sales_trend,
    save_evaluation_summary,
)
from src.forecasting import generate_forecast, save_forecast_csv
from src import database as db


def main():
    logger.info("=" * 60)
    logger.info("E-commerce Sales Prediction & Analytics — Pipeline Start")
    logger.info("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────────
    logger.info("\n[1/9] Loading raw dataset …")
    try:
        raw_df = load_raw_data(config.DATASET_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error(
            "\nPlease download 'Online Retail.xlsx' from:\n"
            "  https://archive.ics.uci.edu/dataset/352/online+retail\n"
            "and place it in: %s", config.DATA_DIR
        )
        sys.exit(1)

    summary = dataset_summary(raw_df)
    logger.info("Raw dataset summary: %s", json.dumps(summary, indent=2))

    # ── 2. Clean data ─────────────────────────────────────────────
    logger.info("\n[2/9] Cleaning transactions …")
    clean_df = clean_transactions(raw_df)
    clean_sum = cleaning_summary(raw_df, clean_df)
    logger.info("Cleaning summary: %s", json.dumps(clean_sum, indent=2))

    # ── 3. Aggregate to daily ─────────────────────────────────────
    logger.info("\n[3/9] Aggregating to daily revenue …")
    daily_df = aggregate_daily_revenue(clean_df)

    # ── 4. MongoDB import ─────────────────────────────────────────
    logger.info("\n[4/9] Importing to MongoDB …")
    db_status = db.health_check()
    if db_status["ok"]:
        db.init_db()
        records = daily_df.copy()
        records["date"] = records["date"].astype(str)
        result = db.insert_sales_records(records.to_dict("records"))
        logger.info("MongoDB import: %s", result)
    else:
        logger.warning("MongoDB unavailable: %s — skipping DB import.", db_status["message"])

    # ── 5. Feature engineering ────────────────────────────────────
    logger.info("\n[5/9] Building features …")
    feat_df = build_features(daily_df)
    train_df, val_df, test_df = chronological_split(feat_df)
    feature_cols = get_feature_columns(feat_df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── 6. Regression ─────────────────────────────────────────────
    logger.info("\n[6/9] Training regression models …")
    reg_metrics = run_all_regression(train_df, val_df, test_df, feature_cols)
    save_regression_metrics_csv(reg_metrics, config.REGRESSION_METRICS_CSV)
    plot_model_comparison(reg_metrics, config.MODEL_COMPARISON_PNG)
    plot_sales_trend(daily_df, config.SALES_TREND_PNG)

    # Save regression metrics to MongoDB
    if db_status["ok"]:
        for m in reg_metrics:
            db.save_model_metrics({
                "model_name": m["model"],
                "task_type": "regression",
                "r2": m["r2"],
                "mae": m["mae"],
                "version": "1.0",
            })

    # ── 7. Classification ─────────────────────────────────────────
    logger.info("\n[7/9] Training classification models …")
    # Add high_sales target (threshold from train only)
    train_clf, threshold = add_classification_target(train_df, train_df)
    val_clf, _ = add_classification_target(val_df, train_df)
    test_clf, _ = add_classification_target(test_df, train_df)

    clf_metrics = run_all_classification(train_clf, val_clf, test_clf, feature_cols)
    save_classification_metrics_csv(clf_metrics, config.CLASSIFICATION_METRICS_CSV)

    if db_status["ok"]:
        for m in clf_metrics:
            db.save_model_metrics({
                "model_name": m["model"],
                "task_type": "classification",
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "threshold": threshold,
                "version": "1.0",
            })

    # ── 8. Forecast ───────────────────────────────────────────────
    logger.info("\n[8/9] Generating forecast …")
    forecast_df = generate_forecast(daily_df, horizon_days=30)
    save_forecast_csv(forecast_df, config.FORECAST_CSV)

    if db_status["ok"]:
        preds = forecast_df.copy()
        preds["forecast_date"] = preds["date"].astype(str)
        db.save_predictions(preds.to_dict("records"))

    # ── 9. Save summary JSON ──────────────────────────────────────
    logger.info("\n[9/9] Saving evaluation summary …")
    save_evaluation_summary(reg_metrics, clf_metrics, config.EVALUATION_SUMMARY_JSON)

    # ── Done ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete! Results saved to: %s", config.RESULTS_DIR)
    logger.info("=" * 60)

    logger.info("\nRegression Results:")
    for m in reg_metrics:
        logger.info("  %-30s  R²=%.4f  MAE=£%.0f", m["model"], m["r2"], m["mae"])

    logger.info("\nClassification Results:")
    for m in clf_metrics:
        logger.info("  %-30s  Acc=%.4f  F1=%.4f", m["model"], m["accuracy"], m["f1"])

    logger.info("\nNext steps:")
    logger.info("  streamlit run app.py          — launch the dashboard")
    logger.info("  python reports/generate_report.py  — generate PDF report")
    logger.info("  pytest -q                     — run tests")


if __name__ == "__main__":
    main()
