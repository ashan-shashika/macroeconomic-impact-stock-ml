def r2_comparison_plot(eval_results):
    """
    Grouped bar chart comparing Train and Test R² across models.

    Parameters
    ----------
    eval_results : dict
        {model_name: (train_m, test_m, train_pred, test_pred)}
        Direct output from evaluate_model().

    Example
    -------
    ev_base = evaluate_model(baseline_model, dtrain, y_train, dtest, y_test, "Baseline")
    ev_opt  = evaluate_model(opt_model,      dtrain, y_train, dtest, y_test, "Optimised")
    ev_tune = evaluate_model(tuned_model,    dtrain, y_train, dtest, y_test, "Tuned")

    plot_r2_comparison({
        "Baseline":  ev_base,
        "Optimised": ev_opt,
        "Tuned":     ev_tune,
    })
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(eval_results.keys())
    train_r2 = [v[0]['r2'] for v in eval_results.values()]
    test_r2 = [v[1]['r2'] for v in eval_results.values()]

    x = np.arange(len(names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_train = ax.bar(x - width/2, train_r2, width, label='Train R²',
                        color='#3498db', edgecolor='black', linewidth=0.5)
    bars_test = ax.bar(x + width/2, test_r2,  width, label='Test R²',
                       color='#e74c3c', edgecolor='black', linewidth=0.5)

    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#3498db')

    for bar in bars_test:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Model Comparison — Train vs Test R²',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(train_r2) + 0.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)

    fig.tight_layout()
    plt.show()
