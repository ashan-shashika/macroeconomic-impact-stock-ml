"""
This module provides a function to calculate and display comprehensive
metrics for model evaluation, including R², RMSE, MAE,
and directional accuracy.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def safe_predict(model, x):
    """Return model predictions while supporting both sklearn and Keras-style
    predict methods."""
    try:
        return model.predict(x, verbose=0)
    except TypeError:
        return model.predict(x)


def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name):
    """Compute R², RMSE, MAE, Directional Accuracy for train and test."""

    def metrics(y_true, y_pred):
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'dir': np.mean(np.sign(y_pred.flatten()) ==
                           np.sign(y_true.flatten())) * 100
        }

    train_pred = safe_predict(model, X_tr).flatten()
    test_pred = safe_predict(model, X_te).flatten()

    train_m = metrics(y_tr, train_pred)
    test_m = metrics(y_te, test_pred)

    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*45}")
    print(f"  {'R²':<25} {train_m['r2']:>10.4f} {test_m['r2']:>10.4f}")
    print(
        f"  {'RMSE (%)':<25} {train_m['rmse']:>10.3f} {test_m['rmse']:>10.3f}")
    print(f"  {'MAE (%)':<25} {train_m['mae']:>10.3f} {test_m['mae']:>10.3f}")
    print(
        f"  {'Directional Acc (%)':<25} {train_m['dir']:>10.2f} {test_m['dir']:>10.2f}")

    return train_m, test_m, train_pred, test_pred
