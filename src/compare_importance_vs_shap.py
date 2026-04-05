"""
Model importance (train) vs SHAP importance (test).
"Are features that the model learned actually useful on test data?"
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.config import TRAIN_COLOR, TEST_COLOR


def compare_importance_vs_shap(shap_data, models_obj):
    """
    Compare built-in feature importance (train) vs SHAP (test).

    Agree    → feature is genuinely important
    Disagree → possible overfitting or distribution shift
    """
    model_names = list(shap_data.keys())
    features = shap_data[model_names[0]]["feature_cols"]

    # extract train importance
    train_imp, imp_type = {}, {}

    for name in model_names:
        model = models_obj[name]

        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            imp_type[name] = "Impurity"
        elif hasattr(model, "coef_"):
            raw = np.abs(model.coef_).flatten()
            imp_type[name] = "|Coef|"
        elif hasattr(model, "get_score"):
            fi = model.get_score(importance_type="gain")
            raw = np.array([fi.get(f, 0) for f in features])
            imp_type[name] = "Gain"
        else:
            continue

        train_imp[name] = (raw / raw.sum()) * 100

    # extract SHAP importance
    shap_imp = {}
    for name in model_names:
        vals = shap_data[name]["mean_abs_shap"].reindex(features).values
        shap_imp[name] = (vals / vals.sum()) * 100

    _plot_side_by_side(train_imp, shap_imp, imp_type, model_names, features)
    _plot_rank_shift(train_imp, shap_imp, model_names, features)
    _print_comparison_table(train_imp, shap_imp,
                            imp_type, model_names, features)


def _plot_side_by_side(train_imp, shap_imp, imp_type, model_names, features):
    """Horizontal bars: train importance vs SHAP importance."""

    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        y = np.arange(len(features))
        w = 0.35

        # sort by SHAP
        sort_idx = np.argsort(shap_imp[name])

        ax.barh(y - w/2, train_imp[name][sort_idx], height=w,
                color=TRAIN_COLOR, alpha=0.8,
                label=f"Model ({imp_type[name]})")
        ax.barh(y + w/2, shap_imp[name][sort_idx], height=w,
                color=TEST_COLOR, alpha=0.8,
                label="SHAP (test)")

        ax.set_yticks(y)
        ax.set_yticklabels([features[i] for i in sort_idx], fontsize=10)
        ax.set_xlabel("Importance (%)")
        ax.set_title(name, fontsize=13, fontweight=500)
        ax.legend(frameon=False, fontsize=9)

    plt.suptitle("Model importance (train) vs SHAP importance (test)",
                 fontsize=15, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_rank_shift(train_imp, shap_imp, model_names, features):
    """Slope chart showing rank changes from train to test."""

    n = len(model_names)
    nf = len(features)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        t_rank = np.argsort(np.argsort(-train_imp[name])) + 1
        s_rank = np.argsort(np.argsort(-shap_imp[name])) + 1

        for i, feat in enumerate(features):
            shift = t_rank[i] - s_rank[i]
            if shift > 0:
                color = "#1D9E75"
            elif shift < 0:
                color = "#E24B4A"
            else:
                color = "#888780"

            ax.plot([0, 1], [t_rank[i], s_rank[i]],
                    color=color, linewidth=1.5, alpha=0.7)
            ax.scatter(0, t_rank[i], color=TRAIN_COLOR, s=60, zorder=3,
                       edgecolor="white", linewidth=0.5)
            ax.scatter(1, s_rank[i], color=TEST_COLOR, s=60, zorder=3,
                       edgecolor="white", linewidth=0.5)

            short = feat.replace("_Change", "").replace("_lag", " L")
            ax.text(-0.15, t_rank[i], short,
                    ha="right", va="center", fontsize=9)
            ax.text(1.15, s_rank[i], short, ha="left", va="center", fontsize=9)

        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(nf + 0.5, 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Model\n(train)", "SHAP\n(test)"], fontsize=11)
        ax.set_ylabel("Rank (1 = most important)")
        ax.set_title(name, fontsize=13, fontweight=500)
        ax.margins(x=0.3)

    plt.suptitle("Feature rank shift: train → test",
                 fontsize=15, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()


def _print_comparison_table(train_imp, shap_imp, imp_type, model_names,
                            features):
    """Print detailed comparison table with CHECK flags."""

    print(f"\n{'':=<80}")
    print("Feature importance: model (train) vs SHAP (test)")
    print(f"{'':=<80}")

    for name in model_names:
        t_rank = np.argsort(np.argsort(-train_imp[name])) + 1
        s_rank = np.argsort(np.argsort(-shap_imp[name])) + 1

        print(f"\n  {name} ({imp_type[name]})")
        print(f"  {'Feature':<25} {'Train%':>7} {'Rank':>5} "
              f"{'SHAP%':>7} {'Rank':>5} {'Shift':>6} {'Flag':>6}")
        print(f"  {'-'*62}")

        for i, feat in enumerate(features):
            shift = t_rank[i] - s_rank[i]
            flag = "CHECK" if shift < -1 else ""
            print(f"  {feat:<25} {train_imp[name][i]:>6.1f}% {t_rank[i]:>4d} "
                  f"{shap_imp[name][i]:>6.1f}% {s_rank[i]:>4d} "
                  f"{shift:>+5d}  {flag:>5}")

    print("\n  Positive shift = ranks higher on test (good)")
    print("  CHECK = dropped 2+ ranks on test (possible overfit)")
    print(f"{'':=<80}")
