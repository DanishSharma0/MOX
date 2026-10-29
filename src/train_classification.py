"""
src/train_classification.py
────────────────────────────
Train and evaluate binary classification models.

Target: high_sales (1) if daily revenue >= training-set median, else (0).
Threshold is derived from training data ONLY to prevent leakage.

Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.evaluation import evaluate_classification, plot_confusion_matrix

logger = logging.getLogger(__name__)


def _xy_clf(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].values
    y = df["high_sales"].values
    return X, y


def save_model(model, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Classifier saved → %s", filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    X_train, y_train = _xy_clf(train_df, feature_cols)
    X_eval, y_eval = _xy_clf(eval_df, feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_eval_s)

    metrics = evaluate_classification(y_eval, y_pred)
    metrics["model"] = "Logistic Regression"
    logger.info("Logistic Regression clf → %s", metrics)

    save_model(model, config.LR_CLASSIFIER_PATH)
    save_model(scaler, config.SCALER_PATH)
    return metrics, y_eval, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

def train_decision_tree(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    X_train, y_train = _xy_clf(train_df, feature_cols)
    X_eval, y_eval = _xy_clf(eval_df, feature_cols)

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = evaluate_classification(y_eval, y_pred)
    metrics["model"] = "Decision Tree"
    logger.info("Decision Tree clf → %s", metrics)

    save_model(model, config.DT_CLASSIFIER_PATH)
    return metrics, y_eval, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest_classifier(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    X_train, y_train = _xy_clf(train_df, feature_cols)
    X_eval, y_eval = _xy_clf(eval_df, feature_cols)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = evaluate_classification(y_eval, y_pred)
    metrics["model"] = "Random Forest Classifier"
    logger.info("Random Forest clf → %s", metrics)

    save_model(model, config.RF_CLASSIFIER_PATH)
    return metrics, y_eval, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Run all classifiers
# ─────────────────────────────────────────────────────────────────────────────

def run_all_classification(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    """
    Run all three classifiers.  Evaluation on val+test.
    Saves confusion matrix for best model (Random Forest).
    """
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    metrics_list = []

    lr_m, _, _ = train_logistic_regression(train_df, eval_df, feature_cols)
    dt_m, _, _ = train_decision_tree(train_df, eval_df, feature_cols)
    rf_m, y_true, y_pred = train_random_forest_classifier(train_df, eval_df, feature_cols)

    metrics_list.extend([lr_m, dt_m, rf_m])

    # Confusion matrix for the best model (RF)
    plot_confusion_matrix(y_true, y_pred, "Random Forest Classifier",
                          config.CONFUSION_MATRIX_PNG)

    return metrics_list
