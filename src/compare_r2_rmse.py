"""
R² and RMSE bar charts.
How do they compare visually?
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.config import TRAIN_COLOR, TEST_COLOR


def compare_r2(results):
    """Side-by-side R² comparison: train vs test."""

    model_names = list(results.keys())
    train_r2 = [results[m]["train_metrics"]["r2"] for m in model_names]
    test_r2 = [results[m]["test_metrics"]["r2"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    w = 0.30

    b1 = ax.bar(x - w/2, train_r2, w, label="Train",
                color=TRAIN_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, test_r2, w, label="Test",
                color=TEST_COLOR, alpha=0.85)

    for bar, val in zip(b1, train_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", fontsize=9, fontweight=500)
    for bar, val in zip(b2, test_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", fontsize=9, fontweight=500)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("R² score")
    ax.set_title("R² comparison across models", fontsize=14, fontweight=500)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(train_r2 + test_r2) * 1.25)
    plt.tight_layout()
    plt.show()


def compare_rmse(results):
    """Side-by-side RMSE comparison: train vs test."""

    model_names = list(results.keys())
    train_rmse = [results[m]["train_metrics"]["rmse"] for m in model_names]
    test_rmse = [results[m]["test_metrics"]["rmse"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    w = 0.30

    b1 = ax.bar(x - w/2, train_rmse, w, label="Train",
                color=TRAIN_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, test_rmse, w, label="Test",
                color=TEST_COLOR, alpha=0.85)

    for bar, val in zip(b1, train_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=9, fontweight=500)
    for bar, val in zip(b2, test_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=9, fontweight=500)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE comparison across models", fontsize=14, fontweight=500)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(train_rmse + test_rmse) * 1.25)
    plt.tight_layout()
    plt.show()
