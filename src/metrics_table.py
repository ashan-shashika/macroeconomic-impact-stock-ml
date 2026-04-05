"""
Metrics overview table.
Which model fits best?
"""


def print_metrics_table(results):
    """Print a clean summary of all metrics across models."""

    header = (f"{'Model':<15} {'R² Train':>10} {'R² Test':>10} "
              f"{'RMSE Train':>12} {'RMSE Test':>12} "
              f"{'MAE Train':>10} {'MAE Test':>10} "
              f"{'Dir Train':>10} {'Dir Test':>10}")

    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for name, r in results.items():
        tr = r["train_metrics"]
        te = r["test_metrics"]
        print(f"{name:<15} "
              f"{tr['r2']:>10.4f} {te['r2']:>10.4f} "
              f"{tr['rmse']:>12.3f} {te['rmse']:>12.3f} "
              f"{tr['mae']:>10.3f} {te['mae']:>10.3f} "
              f"{tr['dir']:>10.2f} {te['dir']:>10.2f}")

    print("=" * len(header))
