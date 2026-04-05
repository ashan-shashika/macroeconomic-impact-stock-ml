import os
import pickle
import numpy as np


def save_evaluation(
    model_name,
    ev_result,
    feature_cols,
    y_test,
    base_dir="models",
):
    """Save model evaluation results to a file.
    Stores training and testing metrics, predictions, actual test values,
    and feature names as a pickle file for later analysis."""
    save_dir = os.path.join(base_dir, "results")
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

    print(f"  Saved evaluation  → {path}")
