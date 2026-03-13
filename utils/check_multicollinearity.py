"""
Utility functions for checking multicollinearity using Variance
Inflation Factor (VIF).
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def check_multicollinearity(feature_matrix: pd.DataFrame,
                            threshold: float = 5.0):
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=5.0
        VIF threshold for concern (typically 5 or 10)

    Returns:
    --------
    pd.DataFrame
        VIF values for each feature

    Example:
    --------
    vif = check_multicollinearity(X_train_df)
    print(vif)
    """

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_matrix.columns
    vif_data["VIF"] = [variance_inflation_factor(
        feature_matrix.values, i) for i in range(len(feature_matrix.columns))]

    vif_data['Concern'] = vif_data['VIF'].apply(
        lambda x: 'High' if x > threshold else 'OK'
    )

    vif_data = vif_data.sort_values('VIF', ascending=False)
    print("\nVariance Inflation Factor (VIF):")
    print("=" * 50)
    print(vif_data.to_string(index=False))
    print("=" * 50)
    print(f"\nNote: VIF > {threshold} indicates multicollinearity concern")

    return vif_data
