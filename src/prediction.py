"""
Prediction & residual analysis.
What are models actually predicting, and where do they fail?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils.config import get_color


def compare_predictions(results, y_test):
    """
    Residual analysis: actual vs predicted,
    violin plot, cumulative error, and error percentiles.
    """
    model_names = list(results.keys())
    colors = [get_color(m) for m in model_names]

    # compute residuals
    residuals = {m: y_test - results[m]["test_pred"] for m in model_names}

    _plot_actual_vs_predicted(results, y_test, model_names, colors)
    _plot_residual_violin(residuals, model_names, colors)
    # _plot_cumulative_error(residuals, model_names, colors)
    # _plot_error_percentiles(residuals, model_names, colors)
    _print_residual_stats(residuals, model_names)


def _plot_actual_vs_predicted(results, y_test, model_names, colors):
    """Scatter: actual vs predicted with fit line."""

    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        pred = results[name]["test_pred"]

        ax.scatter(y_test, pred, alpha=0.5, s=30, color=color,
                   edgecolor="white", linewidth=0.3)

        # perfect line
        lims = [min(y_test.min(), pred.min()) - 1,
                max(y_test.max(), pred.max()) + 1]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="Perfect")

        # fit line
        z = np.polyfit(y_test, pred, 1)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_line, np.polyval(z, x_line), color=color, linewidth=2,
                label=f"Fit (slope={z[0]:.2f})")

        r2 = results[name]["test_metrics"]["r2"]
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight=500, color=color)

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name, fontsize=13, fontweight=500)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    plt.suptitle("Actual vs predicted",
                 fontsize=15, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_residual_violin(residuals, model_names, colors):
    """Violin + strip plot of residuals."""

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [residuals[m] for m in model_names]
    parts = ax.violinplot(data, positions=range(len(model_names)),
                          showmeans=True, showmedians=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.25)
        pc.set_edgecolor(colors[i])
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")

    # scatter overlay
    for i, (name, color) in enumerate(zip(model_names, colors)):
        jitter = np.random.normal(0, 0.05, len(residuals[name]))
        ax.scatter(i + jitter, residuals[name], alpha=0.4, s=15, color=color)

    # annotate bias and std
    for i, name in enumerate(model_names):
        r = residuals[name]
        ax.text(i, ax.get_ylim()[1] * 0.85,
                f"bias={np.mean(r):.2f}\nstd={np.std(r):.2f}",
                ha="center", fontsize=9, color=colors[i], fontweight=500)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residual distribution",
                 fontsize=14, fontweight=500)
    plt.tight_layout()
    plt.show()


def _plot_cumulative_error(residuals, model_names, colors):
    """Cumulative error distribution — steeper = better."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, color in zip(model_names, colors):
        abs_err = np.sort(np.abs(residuals[name]))
        cumulative = np.arange(1, len(abs_err) + 1) / len(abs_err) * 100
        ax.plot(abs_err, cumulative, color=color, linewidth=2.5, label=name)

        # mark P80
        p80 = np.percentile(np.abs(residuals[name]), 80)
        ax.plot(p80, 80, "o", color=color, markersize=7, zorder=5)
        # ax.annotate(f"{p80:.2f}", (p80, 80), textcoords="offset points",
        #             xytext=(10, -5), fontsize=9, color=color, fontweight=500)

    ax.axhline(80, color="gray", linewidth=0.5, ls=":", alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.8, 82, "80th percentile",
            fontsize=9, color="gray")

    ax.set_xlabel("Absolute error")
    ax.set_ylabel("% of predictions below this error")
    ax.set_title("Cumulative error distribution — steeper = better",
                 fontsize=14, fontweight=500)
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(0, 102)
    plt.tight_layout()
    plt.show()


def _plot_error_percentiles(residuals, model_names, colors):
    """Error at each percentile — where do models diverge?"""

    percentiles = [10, 25, 50, 75, 90, 95]
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(percentiles))
    n = len(model_names)
    w = 0.7 / n

    for i, (name, color) in enumerate(zip(model_names, colors)):
        abs_err = np.abs(residuals[name])
        vals = [np.percentile(abs_err, p) for p in percentiles]
        bars = ax.bar(x + i * w, vals, w, color=color, alpha=0.8, label=name)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight=500)

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels([f"P{p}" for p in percentiles])
    ax.set_ylabel("Absolute error")
    ax.set_title("Error at each percentile — where do models diverge?",
                 fontsize=14, fontweight=500)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def _print_residual_stats(residuals, model_names):
    """Print residual diagnostic table."""

    print(f"\n{'Model':<15} {'Bias':>8} {'Std':>8} {'Skew':>8} "
          f"{'P50 err':>8} {'P80 err':>8} {'P95 err':>8}")
    print("=" * 65)

    for name in model_names:
        r = residuals[name]
        ae = np.abs(r)
        print(f"{name:<15} {np.mean(r):>8.3f} {np.std(r):>8.3f} "
              f"{stats.skew(r):>8.3f} "
              f"{np.percentile(ae, 50):>8.3f} "
              f"{np.percentile(ae, 80):>8.3f} "
              f"{np.percentile(ae, 95):>8.3f}")
