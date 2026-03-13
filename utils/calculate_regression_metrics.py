"""
This module provides a function to calculate and display comprehensive
regression metrics for model evaluation, including R², RMSE, MAE,
and directional accuracy.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_regression_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "Model"):
    """
    Calculate comprehensive regression metrics.

    Metrics:
    - R² (R-squared): Proportion of variance explained
    - RMSE (Root Mean Squared Error): In percentage points
    - MAE (Mean Absolute Error): In percentage points
    - Directional Accuracy: % of correct direction predictions

    Parameters:
    -----------
    y_true : np.ndarray
        True values (actual returns)
    y_pred : np.ndarray
        Predicted values
    model_name : str, default="Model"
        Name of model for display

    Returns:
    --------
    Dict[str, float]
        Dictionary with metric names and values

    Example:
    --------
    metrics = calculate_regression_metrics(y_test, predictions, "XGBoost")
    print(f"R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.2f}%")
    """

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Directional accuracy
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    dir_accuracy = np.mean(true_direction == pred_direction) * 100

    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Directional_Accuracy': dir_accuracy
    }

    # Print formatted results
    print(f"\n{model_name} Performance:")
    print(f"  R²:                  {r2:.4f}")
    print(f"  RMSE:                {rmse:.3f}%")
    print(f"  MAE:                 {mae:.3f}%")
    print(f"  Directional Accuracy: {dir_accuracy:.2f}%")

    return metrics
