"""
config.py — Central configuration for the project.
Reads environment variables from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ───────────────────────────────────────────────
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("DB_NAME", "ecommerce_sales_ml")

# ── Collection names ──────────────────────────────────────
SALES_COLLECTION = "sales"
PREDICTIONS_COLLECTION = "predictions"
METRICS_COLLECTION = "model_metrics"

# ── File paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DATASET_PATH = os.path.join(DATA_DIR, "Online Retail.xlsx")

# ── Model file names ──────────────────────────────────────
LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.joblib")
RF_REGRESSOR_PATH = os.path.join(MODELS_DIR, "random_forest_regressor.joblib")
LR_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "logistic_regression_clf.joblib")
DT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "decision_tree_clf.joblib")
RF_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "random_forest_clf.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# ── Result file names ─────────────────────────────────────
REGRESSION_METRICS_CSV = os.path.join(RESULTS_DIR, "regression_metrics.csv")
CLASSIFICATION_METRICS_CSV = os.path.join(RESULTS_DIR, "classification_metrics.csv")
FORECAST_CSV = os.path.join(RESULTS_DIR, "forecast_predictions.csv")
MODEL_COMPARISON_PNG = os.path.join(RESULTS_DIR, "model_comparison.png")
SALES_TREND_PNG = os.path.join(RESULTS_DIR, "sales_trend.png")
CONFUSION_MATRIX_PNG = os.path.join(RESULTS_DIR, "confusion_matrix.png")
EVALUATION_SUMMARY_JSON = os.path.join(RESULTS_DIR, "evaluation_summary.json")
REPORT_PDF = os.path.join(REPORTS_DIR, "project_report.pdf")

# ── Feature engineering settings ──────────────────────────
LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is implicitly 0.15

# ── Forecast settings ─────────────────────────────────────
FORECAST_HORIZON_DEFAULT = 30  # days

# Create directories if they don't exist
for _dir in [DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
