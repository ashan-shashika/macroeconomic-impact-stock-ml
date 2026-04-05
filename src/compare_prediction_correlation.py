"""
Prediction correlation.
Do models agree on individual predictions?
High correlation = redundant. Low correlation = ensembling could help.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.config import get_color


def compare_prediction_correlation(results):
    """Scatter matrix of model predictions + correlation values."""

    model_names = list(results.keys())
    n = len(model_names)
    colors = [get_color(m) for m in model_names]

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            ax = axes[i][j]

            if i == j:
                # diagonal: histogram
                ax.hist(results[m1]["test_pred"], bins=20,
                        color=colors[i], alpha=0.6)
                ax.set_title(m1, fontsize=11, fontweight=500)
            else:
                # off-diagonal: scatter
                ax.scatter(results[m2]["test_pred"],
                           results[m1]["test_pred"],
                           alpha=0.4, s=15, color=colors[i])

                corr = np.corrcoef(results[m1]["test_pred"],
                                   results[m2]["test_pred"])[0, 1]
                ax.set_title(f"r = {corr:.3f}", fontsize=10)

                # diagonal line
                lims = [min(results[m2]["test_pred"].min(),
                            results[m1]["test_pred"].min()),
                        max(results[m2]["test_pred"].max(),
                            results[m1]["test_pred"].max())]
                ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)

            if j == 0:
                ax.set_ylabel(m1, fontsize=10)
            if i == n - 1:
                ax.set_xlabel(m2, fontsize=10)

    plt.suptitle("Prediction correlation matrix (test set)",
                 fontsize=14, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()

    # print correlation table
    print("\nPrediction correlation (test set)")
    print("=" * 50)
    print(f"{'':>15}", end="")
    for m in model_names:
        print(f"{m:>12}", end="")
    print()

    for m1 in model_names:
        print(f"{m1:>15}", end="")
        for m2 in model_names:
            corr = np.corrcoef(results[m1]["test_pred"],
                               results[m2]["test_pred"])[0, 1]
            print(f"{corr:>12.3f}", end="")
        print()
    print("=" * 50)
