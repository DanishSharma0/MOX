"""
tests/test_evaluation.py — Tests for evaluation metrics module.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.evaluation import evaluate_regression, evaluate_classification


def test_perfect_regression():
    y = np.array([100, 200, 300, 400])
    metrics = evaluate_regression(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_regression_r2_negative_possible():
    """A terrible model can have negative R²."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([300, 100, 200])
    metrics = evaluate_regression(y_true, y_pred)
    assert "r2" in metrics
    assert "mae" in metrics


def test_regression_mae_formula():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = evaluate_regression(y_true, y_pred)
    expected_mae = (10 + 10 + 10) / 3
    assert metrics["mae"] == pytest.approx(expected_mae, rel=1e-3)


def test_perfect_classification():
    y = np.array([0, 1, 1, 0, 1])
    metrics = evaluate_classification(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_all_wrong():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    metrics = evaluate_classification(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_classification_keys():
    y = np.array([0, 1])
    metrics = evaluate_classification(y, y)
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in metrics


def test_regression_keys():
    y = np.array([1.0, 2.0])
    metrics = evaluate_regression(y, y)
    for key in ["r2", "mae"]:
        assert key in metrics


def test_regression_constant_prediction():
    """Predicting the mean gives R²=0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full(5, y_true.mean())
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-6)
