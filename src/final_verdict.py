"""
Final verdict.
Which model should we use?
"""


def print_final_verdict(results, shap_data):
    """Summary table with clear recommendation based on all evidence."""

    model_names = list(results.keys())

    print("\n" + "=" * 65)
    print(f"{'FINAL MODEL COMPARISON':^65}")
    print("=" * 65)

    # find best per criterion
    best_r2 = max(model_names,
                  key=lambda m: results[m]["test_metrics"]["r2"])
    best_rmse = min(model_names,
                    key=lambda m: results[m]["test_metrics"]["rmse"])
    best_mae = min(model_names,
                   key=lambda m: results[m]["test_metrics"]["mae"])
    best_dir = max(model_names,
                   key=lambda m: results[m]["test_metrics"]["dir"])
    least_overfit = min(model_names,
                        key=lambda m: (results[m]["train_metrics"]["r2"]
                                       - results[m]["test_metrics"]["r2"]))

    print(f"\n  {'Criterion':<30} {'Winner':<15} {'Value':<15}")
    print(f"  {'-'*55}")

    rows = [
        ("Best test R²", best_r2,
         f"{results[best_r2]['test_metrics']['r2']:.4f}"),
        ("Best test RMSE", best_rmse,
         f"{results[best_rmse]['test_metrics']['rmse']:.3f}"),
        ("Best test MAE", best_mae,
         f"{results[best_mae]['test_metrics']['mae']:.3f}"),
        ("Best direction accuracy", best_dir,
         f"{results[best_dir]['test_metrics']['dir']:.2f}%"),
        ("Least overfitting", least_overfit, ""),
    ]

    for label, winner, value in rows:
        print(f"  {label:<30} {winner:<15} {value:<15}")

    # top SHAP feature per model
    print(f"\n  {'Model':<15} {'Top SHAP feature':<25} {'SHAP value':<12}")
    print(f"  {'-'*50}")
    for name in model_names:
        top = shap_data[name]["mean_abs_shap"].idxmax()
        top_val = shap_data[name]["mean_abs_shap"].max()
        print(f"  {name:<15} {top:<25} {top_val:<12.4f}")

    # count wins per model
    wins = {m: 0 for m in model_names}
    for winner in [best_r2, best_rmse, best_mae, best_dir, least_overfit]:
        wins[winner] += 1

    print(f"\n  {'Model':<15} {'Wins':>6}")
    print(f"  {'-'*22}")
    for name in model_names:
        bar = "█" * wins[name]
        print(f"  {name:<15} {wins[name]:>4}  {bar}")

    overall = max(wins, key=wins.get)
    print(f"\n  Recommendation: {overall} ({wins[overall]}/5 criteria)")
    print("=" * 65)
