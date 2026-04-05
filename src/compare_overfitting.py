"""
Overfitting gap analysis.
"Are any models memorising noise?"
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_overfitting(results):
    """Heatmap showing train-to-test degradation per metric."""

    model_names = list(results.keys())

    gap_data = pd.DataFrame({
        "R² drop": [
            results[m]["train_metrics"]["r2"] -
            results[m]["test_metrics"]["r2"]
            for m in model_names],
        "RMSE increase": [
            results[m]["test_metrics"]["rmse"] -
            results[m]["train_metrics"]["rmse"]
            for m in model_names],
        "MAE increase": [
            results[m]["test_metrics"]["mae"] -
            results[m]["train_metrics"]["mae"]
            for m in model_names],
        "Dir acc drop": [
            results[m]["train_metrics"]["dir"] -
            results[m]["test_metrics"]["dir"]
            for m in model_names],
    }, index=model_names)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(gap_data, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"fontsize": 11, "fontweight": 500})
    ax.set_title("Overfitting Sensitivity Heatmap",
                 fontsize=14, fontweight=500, pad=12)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.show()
