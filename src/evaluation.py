"""
src/evaluation.py
─────────────────
Compute and save evaluation metrics for regression and classification tasks.
"""
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with r2 and mae."""
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": round(r2, 4), "mae": round(mae, 2)}


def save_regression_metrics_csv(metrics_list: list[dict], filepath: str) -> None:
    """Save a list of {model, r2, mae} dicts to CSV."""
    pd.DataFrame(metrics_list).to_csv(filepath, index=False)
    logger.info("Regression metrics saved → %s", filepath)


def plot_model_comparison(metrics_list: list[dict], filepath: str) -> None:
    """Bar chart comparing R² and MAE across models."""
    df = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")

    # R² — higher is better
    axes[0].bar(df["model"], df["r2"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("R² Score (higher = better)")
    axes[0].set_ylabel("R² Score")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(df["r2"]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    # MAE — lower is better
    axes[1].bar(df["model"], df["mae"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("MAE (lower = better)")
    axes[1].set_ylabel("Mean Absolute Error (£)")
    for i, v in enumerate(df["mae"]):
        axes[1].text(i, v + max(df["mae"]) * 0.01, f"£{v:.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Model comparison chart saved → %s", filepath)


def plot_sales_trend(daily_df: pd.DataFrame, filepath: str) -> None:
    """Line chart of historical daily revenue."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_df["date"], daily_df["revenue"], color="#4C72B0", linewidth=0.9, alpha=0.8)
    ax.fill_between(daily_df["date"], daily_df["revenue"], alpha=0.15, color="#4C72B0")
    ax.set_title("Historical Daily Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (£)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Sales trend chart saved → %s", filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with accuracy, precision, recall, f1."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def save_classification_metrics_csv(metrics_list: list[dict], filepath: str) -> None:
    pd.DataFrame(metrics_list).to_csv(filepath, index=False)
    logger.info("Classification metrics saved → %s", filepath)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filepath: str,
) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Low Sales (0)", "High Sales (1)"],
        yticklabels=["Low Sales (0)", "High Sales (1)"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Summary JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_summary(
    regression_metrics: list[dict],
    classification_metrics: list[dict],
    filepath: str,
) -> None:
    summary = {
        "regression": regression_metrics,
        "classification": classification_metrics,
    }
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation summary saved → %s", filepath)
