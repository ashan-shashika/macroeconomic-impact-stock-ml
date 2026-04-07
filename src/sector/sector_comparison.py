"""
Compare feature importance across sectors.
The main deliverable: which macro features drive which sectors?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

SECTOR_NAMES = {
    "tech":       "Technology",
    "healthcare": "Healthcare",
    "finance":    "Financials",
    "industrial": "Industrials",
    "energy":     "Energy",
}

FEATURE_LABELS = {
    "VIX_Change":         "VIX (fear)",
    "Rate_Change":        "Interest rates",
    "USD_Change":         "US Dollar",
    "GDP_Growth_lag2":    "GDP growth",
    "CPI_Change_lag1":    "Inflation",
    "Unemp_Change_lag1":  "Unemployment",
    "Credit_Spread_lag2": "Credit spreads",
}


def build_sector_shap_matrix(shap_data):
    """
    Build a features × sectors matrix of normalised SHAP importance.
    This is the core data structure for all comparison plots.
    """
    sectors = list(shap_data.keys())
    features = shap_data[sectors[0]]["feature_cols"]

    matrix = pd.DataFrame(index=features, columns=sectors)
    for sector in sectors:
        vals = shap_data[sector]["mean_abs_shap"].reindex(features)
        matrix[sector] = (vals / vals.sum()) * 100

    matrix = matrix.astype(float)
    return matrix


def plot_sector_heatmap(shap_data, ax=None):
    standalone = ax is None

    matrix = build_sector_shap_matrix(shap_data)

    matrix.index = [FEATURE_LABELS.get(f, f) for f in matrix.index]
    matrix.columns = [SECTOR_NAMES.get(s, s) for s in matrix.columns]

    matrix = matrix = matrix.sort_index()

    if standalone:
        fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"fontsize": 10, "fontweight": 500},
        cbar_kws={"label": "% of total SHAP"}
    )

    ax.set_title("Which macro features drive which sectors?",
                 fontsize=15, fontweight=500, pad=14)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel("")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_sector_feature_profiles(shap_data):
    """
    Radar chart per sector showing its feature sensitivity profile.
    Sectors with similar shapes react to the same macro forces.
    """
    from math import pi

    matrix = build_sector_shap_matrix(shap_data)
    features = list(matrix.index)
    sectors = list(matrix.columns)
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    # group into 2 rows for readability
    n = len(sectors)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    angles = [i / len(features) * 2 * pi for i in range(len(features))]
    angles += angles[:1]

    for i, sector in enumerate(sectors):
        ax = axes[i]
        vals = matrix[sector].values.tolist()
        vals += [vals[0]]

        ax.plot(angles, vals, linewidth=2, color="#D85A30")
        ax.fill(angles, vals, alpha=0.15, color="#D85A30")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_title(SECTOR_NAMES.get(sector, sector),
                     fontsize=12, fontweight=500, pad=15)

    # hide unused subplots
    for j in range(len(sectors), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Sector sensitivity profiles",
                 fontsize=15, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_across_sectors(shap_data, feature="VIX_Change"):
    """
    Bar chart: how important is ONE feature across all sectors?
    e.g. "Which sectors are most sensitive to VIX?"
    """
    matrix = build_sector_shap_matrix(shap_data)
    # sectors = list(matrix.columns)
    vals = matrix.loc[feature]

    # sort by importance
    sorted_idx = vals.sort_values(ascending=True).index
    sorted_vals = vals[sorted_idx]
    sorted_names = [SECTOR_NAMES.get(s, s) for s in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = mpl.colormaps["YlOrRd"]
    colors = cmap(sorted_vals / sorted_vals.max())
    ax.barh(range(len(sorted_vals)), sorted_vals, color=colors, height=0.6)

    for i, val in enumerate(sorted_vals):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center",
                fontsize=10, fontweight=500)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("SHAP importance (%)")

    plain = FEATURE_LABELS.get(feature, feature)
    ax.set_title(f"Which sectors are most sensitive to {plain}?",
                 fontsize=14, fontweight=500)
    plt.tight_layout()
    plt.show()


def plot_sector_clustering(shap_data):
    """
    Clustermap: group sectors by similar feature sensitivity.
    Sectors that cluster together react to the same macro forces.
    """
    matrix = build_sector_shap_matrix(shap_data)
    matrix.index = [FEATURE_LABELS.get(f, f) for f in matrix.index]
    matrix.columns = [SECTOR_NAMES.get(s, s) for s in matrix.columns]

    g = sns.clustermap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                       linewidths=0.5, figsize=(14, 7),
                       annot_kws={"fontsize": 9, "fontweight": 500},
                       cbar_kws={"label": "% of total SHAP"})
    g.fig.suptitle("Sector clustering by macro sensitivity",
                   fontsize=15, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_directional_impact_by_sector(shap_data):
    """
    For each sector: does VIX/GDP/etc push it UP or DOWN?
    The economic story per sector.
    """
    # UP = "#1D9E75"
    # DOWN = "#E24B4A"

    sectors = list(shap_data.keys())
    features = shap_data[sectors[0]]["feature_cols"]
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    # build direction matrix
    direction = pd.DataFrame(index=features, columns=sectors)
    for sector in sectors:
        shap_df = shap_data[sector]["shap_test_df"]
        for feat in features:
            direction.loc[feat, sector] = np.mean(shap_df[feat].values)

    direction = direction.astype(float)
    direction.index = labels
    direction.columns = [SECTOR_NAMES.get(s, s) for s in sectors]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(direction, annot=True, fmt="+.3f", center=0,
                cmap="RdYlGn", linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"fontsize": 9, "fontweight": 500},
                cbar_kws={"label": "Green = pushes sector UP"})

    ax.set_title("When this macro factor rises → sector goes...",
                 fontsize=14, fontweight=500, pad=14)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.show()


def print_sector_summary(shap_data):
    """Print the key story: top driver per sector."""

    sectors = list(shap_data.keys())
    # features = shap_data[sectors[0]]["feature_cols"]
    matrix = build_sector_shap_matrix(shap_data)

    print(f"\n{'='*70}")
    print("  TOP MACRO DRIVER PER SECTOR")
    print(f"{'='*70}")
    print(
        f"\n  {'Sector':<20} {'Top feature':<25} "
        f"{'Importance':>12} {'2nd feature':<25}")
    print(f"  {'-'*65}")

    for sector in sectors:
        vals = matrix[sector].sort_values(ascending=False)
        top1 = vals.index[0]
        top2 = vals.index[1]
        name = SECTOR_NAMES.get(sector, sector)
        plain1 = FEATURE_LABELS.get(top1, top1)
        plain2 = FEATURE_LABELS.get(top2, top2)

        print(
            f"  {name:<20} {plain1:<25} {vals.iloc[0]:>10.1f}%  {plain2:<25}")

    print(f"{'='*70}")
