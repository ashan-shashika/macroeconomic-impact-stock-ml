"""
SHAP importance — "Which features matter?"
Feature rank consensus — "Do models agree on what matters?"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from utils.config import get_color


def compare_feature_importance(shap_data):
    """
    Compare feature importance across models using mean |SHAP|.
    """
    model_names = list(shap_data.keys())
    features = shap_data[model_names[0]]["feature_cols"]
    colors = [get_color(m) for m in model_names]

    # build dataframes
    shap_df = pd.DataFrame({
        name: s["mean_abs_shap"] for name, s in shap_data.items()
    })
    shap_norm = shap_df.div(shap_df.sum(axis=0), axis=1) * 100

    _plot_bar(shap_df, model_names, features, colors)
    _plot_heatmap(shap_norm)
    _plot_radar(shap_norm, model_names, features, colors)
    _print_rank_table(shap_df)


def _plot_bar(shap_df, model_names, features, colors):
    """Grouped bar chart of mean |SHAP|."""

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(features))
    n = len(model_names)
    w = 0.7 / n

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = shap_df.loc[features, name].values
        bars = ax.bar(x + i * w, vals, w, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight=500)

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("SHAP feature importance comparison",
                 fontsize=14, fontweight=500)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def _plot_heatmap(shap_norm):
    """Heatmap of normalised SHAP importance."""

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(shap_norm.T, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"fontsize": 11, "fontweight": 500})
    ax.set_title("Feature importance (% of total SHAP)",
                 fontsize=14, fontweight=500, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.show()


def _plot_radar(shap_norm, model_names, features, colors):
    """Radar chart of normalised SHAP importance."""

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    angles = [n / len(features) * 2 * pi for n in range(len(features))]
    angles += angles[:1]

    for name, color in zip(model_names, colors):
        vals = shap_norm.loc[features, name].values.tolist()
        vals += [vals[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=10)
    ax.set_title("Feature importance consensus",
                 fontsize=14, fontweight=500, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    plt.tight_layout()
    plt.show()


def _print_rank_table(shap_df):
    """Print rank table with average rank."""

    rank_df = shap_df.rank(ascending=False).astype(int)
    rank_df["Avg rank"] = rank_df.mean(axis=1).round(1)
    rank_df = rank_df.sort_values("Avg rank")

    print("\nFeature ranking by SHAP (1 = most important)")
    print("=" * 60)
    print(rank_df.to_string())
    print()
