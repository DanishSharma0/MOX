"""
src/database.py
───────────────
All MongoDB operations via PyMongo.
Collections:
  • sales           — cleaned, aggregated historical records
  • predictions     — forecast outputs
  • model_metrics   — evaluation results per model
"""
import sys
import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, ASCENDING, errors as pymongo_errors

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return a cached MongoClient.  Raises ConnectionFailure on error."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        except Exception as exc:
            raise pymongo_errors.ConnectionFailure(
                f"Cannot create MongoClient: {exc}"
            ) from exc
    return _client


def get_db():
    """Return the project database object."""
    return get_client()[config.DB_NAME]


def health_check() -> dict:
    """Ping MongoDB. Returns dict with 'ok' bool and 'message' str."""
    try:
        get_client().admin.command("ping")
        return {"ok": True, "message": "MongoDB is reachable."}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation — create indexes
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create collections and indexes. Safe to call multiple times."""
    db = get_db()

    # sales: unique index on date (daily aggregated)
    db[config.SALES_COLLECTION].create_index(
        [("date", ASCENDING)], unique=True, name="idx_sales_date"
    )

    # predictions: composite index on forecast_date + model_name
    db[config.PREDICTIONS_COLLECTION].create_index(
        [("forecast_date", ASCENDING), ("model_name", ASCENDING)],
        name="idx_pred_date_model",
    )

    # model_metrics: composite index on model_name + task_type + eval_date
    db[config.METRICS_COLLECTION].create_index(
        [("model_name", ASCENDING), ("task_type", ASCENDING), ("eval_date", ASCENDING)],
        name="idx_metric_model_task_date",
    )

    logger.info("Database '%s' initialised.", config.DB_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Sales CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_sales_records(records: list[dict]) -> dict:
    """
    Bulk-insert cleaned & aggregated sales records.
    Skips duplicates via update-or-insert on 'date'.
    Returns {'inserted': int, 'updated': int, 'errors': int}.
    """
    db = get_db()
    col = db[config.SALES_COLLECTION]
    inserted = updated = errors = 0

    for rec in records:
        try:
            result = col.update_one(
                {"date": rec["date"]},
                {"$set": rec},
                upsert=True,
            )
            if result.upserted_id:
                inserted += 1
            else:
                updated += 1
        except Exception as exc:
            logger.warning("Failed to upsert sales record: %s — %s", rec.get("date"), exc)
            errors += 1

    return {"inserted": inserted, "updated": updated, "errors": errors}


def get_all_sales(limit: int = 0) -> list[dict]:
    """Retrieve all sales records sorted by date."""
    col = get_db()[config.SALES_COLLECTION]
    cursor = col.find({}, {"_id": 0}).sort("date", ASCENDING)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def get_daily_revenue() -> list[dict]:
    """Return list of {date, revenue} dicts sorted chronologically."""
    return get_all_sales()


def get_monthly_revenue() -> list[dict]:
    """Aggregate daily revenue to monthly using MongoDB pipeline."""
    pipeline = [
        {
            "$group": {
                "_id": {
                    "year": {"$year": {"$toDate": "$date"}},
                    "month": {"$month": {"$toDate": "$date"}},
                },
                "monthly_revenue": {"$sum": "$revenue"},
                "monthly_quantity": {"$sum": "$quantity"},
            }
        },
        {"$sort": {"_id.year": 1, "_id.month": 1}},
        {
            "$project": {
                "_id": 0,
                "year": "$_id.year",
                "month": "$_id.month",
                "monthly_revenue": 1,
                "monthly_quantity": 1,
            }
        },
    ]
    return list(get_db()[config.SALES_COLLECTION].aggregate(pipeline))


def get_total_stats() -> dict:
    """Return aggregate totals for the dashboard KPIs."""
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_revenue": {"$sum": "$revenue"},
                "total_quantity": {"$sum": "$quantity"},
                "total_transactions": {"$sum": "$num_transactions"},
                "record_count": {"$sum": 1},
            }
        }
    ]
    result = list(get_db()[config.SALES_COLLECTION].aggregate(pipeline))
    if result:
        r = result[0]
        r.pop("_id", None)
        return r
    return {}


def count_sales_records() -> int:
    return get_db()[config.SALES_COLLECTION].count_documents({})


# ─────────────────────────────────────────────────────────────────────────────
# Predictions CRUD
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(predictions: list[dict]) -> dict:
    """
    Insert forecast records.  Each record must include:
      forecast_date, model_name, predicted_revenue, horizon_days, prediction_ts.
    """
    db = get_db()
    col = db[config.PREDICTIONS_COLLECTION]
    inserted = errors = 0
    for pred in predictions:
        pred.setdefault("prediction_ts", datetime.now(timezone.utc).isoformat())
        try:
            col.insert_one(pred)
            inserted += 1
        except Exception as exc:
            logger.warning("Failed to insert prediction: %s", exc)
            errors += 1
    return {"inserted": inserted, "errors": errors}


def get_predictions(model_name: str | None = None, limit: int = 200) -> list[dict]:
    """Retrieve saved predictions, optionally filtered by model name."""
    query: dict[str, Any] = {}
    if model_name:
        query["model_name"] = model_name
    cursor = (
        get_db()[config.PREDICTIONS_COLLECTION]
        .find(query, {"_id": 0})
        .sort("forecast_date", ASCENDING)
        .limit(limit)
    )
    return list(cursor)


# ─────────────────────────────────────────────────────────────────────────────
# Model metrics CRUD
# ─────────────────────────────────────────────────────────────────────────────

def save_model_metrics(metrics: dict) -> bool:
    """Insert one model evaluation result document."""
    metrics.setdefault("eval_date", datetime.now(timezone.utc).isoformat())
    try:
        get_db()[config.METRICS_COLLECTION].insert_one(metrics)
        return True
    except Exception as exc:
        logger.error("Failed to save model metrics: %s", exc)
        return False


def get_model_metrics(task_type: str | None = None) -> list[dict]:
    """Retrieve all saved model metrics, optionally filtered by task_type."""
    query: dict[str, Any] = {}
    if task_type:
        query["task_type"] = task_type
    cursor = (
        get_db()[config.METRICS_COLLECTION]
        .find(query, {"_id": 0})
        .sort("eval_date", ASCENDING)
    )
    return list(cursor)
