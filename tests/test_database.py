"""
tests/test_database.py — Tests for MongoDB operations.
Uses an isolated test database to avoid touching production data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Mocked tests (work without a live MongoDB)
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check_ok(monkeypatch):
    """Simulate a successful ping."""
    from src import database as db
    mock_client = MagicMock()
    mock_client.admin.command.return_value = {"ok": 1}
    monkeypatch.setattr(db, "_client", mock_client)
    result = db.health_check()
    assert result["ok"] is True


def test_health_check_fail(monkeypatch):
    """Simulate a failed connection."""
    from src import database as db
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = Exception("Connection refused")
    monkeypatch.setattr(db, "_client", mock_client)
    result = db.health_check()
    assert result["ok"] is False
    assert "Connection refused" in result["message"]


def test_save_model_metrics_ok(monkeypatch):
    from src import database as db
    mock_col = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    monkeypatch.setattr(db, "get_db", lambda: mock_db)

    metrics = {"model_name": "Test", "task_type": "regression", "r2": 0.9, "mae": 500}
    result = db.save_model_metrics(metrics)
    assert result is True
    mock_col.insert_one.assert_called_once()


def test_get_model_metrics_returns_list(monkeypatch):
    from src import database as db
    fake_records = [
        {"model_name": "LR", "task_type": "regression", "r2": 0.8},
        {"model_name": "RF", "task_type": "regression", "r2": 0.92},
    ]
    mock_col = MagicMock()
    mock_col.find.return_value.sort.return_value = fake_records
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    monkeypatch.setattr(db, "get_db", lambda: mock_db)

    result = db.get_model_metrics()
    assert isinstance(result, list)


def test_insert_sales_records_upserts(monkeypatch):
    from src import database as db
    mock_col = MagicMock()
    upsert_result = MagicMock()
    upsert_result.upserted_id = "new_id"
    mock_col.update_one.return_value = upsert_result
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    monkeypatch.setattr(db, "get_db", lambda: mock_db)

    records = [{"date": "2010-12-01", "revenue": 10000}]
    result = db.insert_sales_records(records)
    assert result["inserted"] == 1
    assert result["errors"] == 0


def test_save_predictions_inserts(monkeypatch):
    from src import database as db
    mock_col = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    monkeypatch.setattr(db, "get_db", lambda: mock_db)

    preds = [{"forecast_date": "2012-01-01", "predicted_revenue": 12000, "model_name": "RF"}]
    result = db.save_predictions(preds)
    assert result["inserted"] == 1


def test_duplicate_prevention(monkeypatch):
    """Inserting the same record twice should result in upsert, not duplicate."""
    from src import database as db
    call_count = {"n": 0}

    def fake_update(query, update, upsert=False):
        call_count["n"] += 1
        m = MagicMock()
        m.upserted_id = "id_1" if call_count["n"] == 1 else None  # 2nd call = update
        return m

    mock_col = MagicMock()
    mock_col.update_one.side_effect = fake_update
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    monkeypatch.setattr(db, "get_db", lambda: mock_db)

    rec = {"date": "2010-12-01", "revenue": 10000}
    r1 = db.insert_sales_records([rec])
    r2 = db.insert_sales_records([rec])
    assert r1["inserted"] == 1
    assert r2["updated"] == 1  # second call is an update, not insert


# ─────────────────────────────────────────────────────────────────────────────
# Live MongoDB tests (skipped if MongoDB is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

import config

TEST_DB_NAME = "ecommerce_test_db"


@pytest.fixture
def live_db():
    """Connect to a temporary test database and clean up after tests."""
    from src import database as db
    hc = db.health_check()
    if not hc["ok"]:
        pytest.skip("MongoDB not available")

    # Use test database
    client = db.get_client()
    test_db = client[TEST_DB_NAME]
    yield test_db
    # Cleanup
    client.drop_database(TEST_DB_NAME)


def test_live_insert_and_retrieve(live_db):
    """Insert a record and retrieve it from MongoDB."""
    col = live_db["test_sales"]
    col.delete_many({})  # clear
    col.insert_one({"date": "2010-12-01", "revenue": 9999.99})
    doc = col.find_one({"date": "2010-12-01"})
    assert doc is not None
    assert doc["revenue"] == 9999.99


def test_live_upsert_no_duplicate(live_db):
    """Upserting the same date twice should not create two documents."""
    col = live_db["test_sales_upsert"]
    col.delete_many({})
    for _ in range(2):
        col.update_one(
            {"date": "2010-12-02"},
            {"$set": {"date": "2010-12-02", "revenue": 5000}},
            upsert=True,
        )
    count = col.count_documents({"date": "2010-12-02"})
    assert count == 1
