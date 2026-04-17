"""
Compare feature importance across sectors.
The main deliverable: which macro features drive which sectors?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

SECTOR_COLOURS = {
    "energy":     "#E07B39",
    "finance":    "#4C72B0",
    "healthcare": "#55A868",
    "industrial": "#C44E52",
    "tech":       "#8172B2",
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


def plot_feature_importance(all_results, model, sectors=None, top_n=None,
                            save_path=None):
    """
    Parameters
    ----------
    all_results : dict    output of load_feature_importances()
    model       : str     e.g. "ridge" | "rf" | "xgboost"
    sectors     : list    e.g. ["energy", "finance"]
                          None → plot all sectors for that model
    top_n       : int     show only top N features
    """
    if model not in all_results:
        raise ValueError(
            f"Model '{model}' not found. "
            f"Available: {list(all_results.keys())}")

    # use all sectors for that model if none specified
    sectors_to_plot = sectors if sectors else list(all_results[model].keys())

    # validate sectors
    for sector in sectors_to_plot:
        if sector not in all_results[model]:
            raise ValueError(f"Sector '{sector}' not found under '{model}'. "
                             f"Available: {list(all_results[model].keys())}")

    n = len(sectors_to_plot)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 4.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, sector in enumerate(sectors_to_plot):
        ax = axes[idx]
        data = all_results[model][sector]

        feature_names = [FEATURE_LABELS.get(f, f)
                         for f in data["feature_cols"]]
        importances = np.array(data["importances_pct"])

        pairs_sorted = sorted(
            zip(importances, feature_names), key=lambda x: x[0])
        if top_n:
            pairs_sorted = pairs_sorted[-top_n:]

        imp_vals, feat_labels = zip(*pairs_sorted)
        imp_vals = np.array(imp_vals)
        colour = SECTOR_COLOURS.get(sector, "#888888")

        y_pos = np.arange(len(feat_labels))
        bars = ax.barh(y_pos, imp_vals, color=colour, alpha=0.85,
                       edgecolor="white", linewidth=0.6)

        x_max = imp_vals.max()
        for bar, val in zip(bars, imp_vals):
            ax.text(val + x_max * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_title(f"{sector.capitalize()}", fontsize=11,
                     fontweight="bold", color=colour)
        ax.set_xlabel("Importance (%)", fontsize=9)
        ax.set_xlim(0, x_max * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model.upper()} — Feature Importance",
                 fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_shap_importance(shap_data, sectors=None, top_n=None, save_path=None):
    """
    Parameters
    ----------
    shap_data : dict    output of load_shap_data()
    sectors   : list    e.g. ["energy", "finance"]
                        None → plot all sectors
    top_n     : int     show only top N features per sector
    save_path : str     optional path to save the figure
    """
    matrix = build_sector_shap_matrix(shap_data)

    if sectors is not None:
        available = [s for s in sectors if s in matrix.columns]
        matrix = matrix[available]

    # Apply display labels
    matrix.index = [FEATURE_LABELS.get(f, f) for f in matrix.index]
    matrix.columns = [SECTOR_NAMES.get(s, s) for s in matrix.columns]

    sectors_to_plot = list(matrix.columns)
    n = len(sectors_to_plot)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 4.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, sector_col in enumerate(sectors_to_plot):
        ax = axes[idx]

        # Sort ascending so highest bar ends up at top after barh
        vals = matrix[sector_col].sort_values(ascending=True)
        if top_n:
            vals = vals.tail(top_n)

        feat_labels = vals.index.tolist()
        imp_vals = vals.values
        y_pos = np.arange(len(feat_labels))

        # Look up colour by original sector key (before SECTOR_NAMES mapping)
        original_key = next(
            (k for k, v in SECTOR_NAMES.items() if v == sector_col), sector_col
        )
        colour = SECTOR_COLOURS.get(original_key, "#888888")

        bars = ax.barh(y_pos, imp_vals, color=colour, alpha=0.85,
                       edgecolor="white", linewidth=0.6)

        x_max = imp_vals.max()
        for bar, val in zip(bars, imp_vals):
            ax.text(val + x_max * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_title(sector_col, fontsize=11, fontweight="bold", color=colour)
        ax.set_xlabel("% of total SHAP", fontsize=9)
        ax.set_xlim(0, x_max * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SHAP — Feature Importance by Sector",
                 fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
