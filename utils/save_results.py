"""
Three functions to save model artifacts.

Folder layout:
    models/
    ├── results/   → metrics + predictions
    ├── shap/      → SHAP values
    └── trained/   → model objects
"""

import os
import pickle
import sys
from pathlib import Path
import numpy as np

root_dir = Path().resolve().parent
sys.path.append(str(root_dir))
model_dir = root_dir / "models"


def save_evaluation(model_name, ev_result, feature_cols, y_test,):
    """
    Save evaluate_model() output.

    Parameters
    model_name : str       e.g. "rf", "ridge", "xgboost"
    ev_result  : tuple     (train_metrics, test_metrics, train_pred, test_pred)
    feature_cols : list    feature column names
    """

    save_dir = os.path.join(model_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    train_metrics, test_metrics, train_pred, test_pred = ev_result

    data = {
        "model_name":    model_name,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "train_pred":    np.array(train_pred),
        "test_pred":     np.array(test_pred),
        "y_test":        np.array(y_test),
        "feature_cols":  feature_cols,
    }

    path = os.path.join(save_dir, f"{model_name}_results.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved evaluation  → {path}")
    print(
        f"R² train={train_metrics['r2']:.4f}  test={test_metrics['r2']:.4f}")


def save_shap(
    model_name, shap_test_df, mean_abs_shap, feature_cols,
):
    """
    Save compute_shap() output.

    Parameters
    model_name    : str
    shap_test_df  : pd.DataFrame   SHAP values from compute_shap()
    mean_abs_shap : pd.Series      mean |SHAP| from compute_shap()
    feature_cols  : list
    """
    save_dir = os.path.join(model_dir, "shap")
    os.makedirs(save_dir, exist_ok=True)

    data = {
        "model_name":    model_name,
        "shap_test_df":  shap_test_df,
        "mean_abs_shap": mean_abs_shap,
        "feature_cols":  feature_cols,
    }

    path = os.path.join(save_dir, f"{model_name}_shap.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"  Saved SHAP        → {path}")
    print(f"    Features: {len(feature_cols)}")


def save_model(model_name, model):
    """
    Save the trained model object.

    Parameters
    model_name : str
    model      : trained model (sklearn, xgboost, keras, etc.)
    """
    save_dir = os.path.join(model_dir, "trained")
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{model_name}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved model       → {path}  ({size_kb:.1f} KB)")
