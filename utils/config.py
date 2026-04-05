"""
Shared configuration: colors, styles, and helper functions.
Import this at the top of every comparison module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── plot style ──
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── colors ──
MODEL_COLORS = {
    "rf":      "#1D9E75",
    "ridge":   "#534AB7",
    "xgboost": "#D85A30",
}
TRAIN_COLOR = "#2D8B75"
TEST_COLOR = "#D85A30"
UP_COLOR = "#1D9E75"
DOWN_COLOR = "#E24B4A"
GRAY = "#888780"


def get_color(name):
    """Get model color, default gray."""
    return MODEL_COLORS.get(name, GRAY)
