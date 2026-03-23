import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def plot_shap_importance(mean_abs_shap):
    """
    Horizontal bar chart of mean absolute SHAP values.
    Shows WHICH features matter most — averaged across all test observations.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    sorted_shap = mean_abs_shap.sort_values()
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(sorted_shap)))

    bars = ax.barh(range(len(sorted_shap)), sorted_shap.values,
                   color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(sorted_shap)))
    ax.set_yticklabels(sorted_shap.index, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title('Feature Importance — Mean |SHAP| on Test Set', fontsize=11)

    # Value labels on bars
    for bar, val in zip(bars, sorted_shap.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_shap_direction(shap_test_df):
    """
    Mean signed SHAP value per feature.
    Unlike mean |SHAP|, this shows NET direction:
    positive bar = feature on average PUSHES predictions UP
    negative bar = feature on average PUSHES predictions DOWN
    (Red = pushes price UP, Blue = pushes price DOWN)
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    mean_signed = shap_test_df.mean().sort_values()
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in mean_signed]

    bars = ax.barh(range(len(mean_signed)), mean_signed.values,
                   color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(mean_signed)))
    ax.set_yticklabels(mean_signed.index, fontsize=10)
    ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Mean Signed SHAP Value (%)', fontsize=11)
    ax.set_title('Feature Direction — Mean Signed SHAP on Test Set',
                 fontsize=11)

    for bar, val in zip(bars, mean_signed.values):
        xpos = bar.get_width() + 0.002 if val >= 0 else bar.get_width() - 0.002
        ha = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}%', va='center', ha=ha, fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')

    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='white', label='Pushes Price Up'),
        Patch(facecolor='#e74c3c', edgecolor='white', label='Pushes Price Down')
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.margins(x=0.15)
    plt.tight_layout()
    plt.show()
