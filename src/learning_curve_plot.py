import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from utils.savePlots import save


def plot_learning_curve(evals_result, model_name="XGBoost"):
    """
    Plot train vs test RMSE learning curve for a single model
    to diagnose overfitting.
    """

    train_rmse = evals_result['train']['rmse']
    test_rmse = evals_result['test']['rmse']
    rounds = np.arange(1, len(train_rmse) + 1)

    # Best test round
    best_idx = int(np.argmin(test_rmse))
    best_test = test_rmse[best_idx]
    best_train = train_rmse[best_idx]
    gap = best_test - best_train

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.5)
    ax.plot(rounds, test_rmse,  label='Test RMSE',  linewidth=1.5)

    ax.fill_between(rounds, train_rmse, test_rmse,
                    alpha=0.15, color='red', label='Overfit gap')

    ax.axvline(x=best_idx + 1, color='grey', linestyle='--',
               alpha=0.6, label=f'Best test round ({best_idx + 1})')
    ax.scatter(best_idx + 1, best_test, color='red', zorder=5, s=60)
    ax.scatter(best_idx + 1, best_train, color='blue', zorder=5, s=60)

    # Annotate gap at best round
    mid_y = (best_train + best_test) / 2

    ax.annotate('', xy=(best_idx + 1, best_train),
                xytext=(best_idx + 1, best_test),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

    ax.text(best_idx + 3, mid_y, f'Gap: {gap:.2f}%',
            fontsize=9, color='red',
            va='center')

    # Fixed y-axis for cross-model comparison
    last_round = rounds[-1]
    ax.set_ylim(0, 7)
    ax.set_xlim(left=0, right=last_round + (50 if best_idx >
                200 else 20 if best_idx > 100 else 10))

    ax.set_title(f'{model_name} — Overfitting Diagnosis',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('RMSE (%)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    print(
        f"Best round: {best_idx + 1}  |  Train RMSE: {best_train:.4f}  | "
        f"Test RMSE: {best_test:.4f}  |  Gap: {gap:.4f}")


def plot_rf_learning_curve(
    rf_model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="Random Forest",
):
    """
    Plot per-tree train vs test RMSE for a fitted Random Forest.
    Returns optimal n_estimators for retraining.
    """
    PLT_SAVE_DIR = "model_RF"
    # cumulative RMSE across trees
    n_trees = len(rf_model.estimators_)
    train_sum = np.zeros(len(y_train))
    test_sum = np.zeros(len(y_test))
    train_rmse, test_rmse = [], []

    for i, tree in enumerate(rf_model.estimators_, 1):
        train_sum += tree.predict(X_train)
        test_sum += tree.predict(X_test)
        train_rmse.append(np.sqrt(mean_squared_error(y_train, train_sum / i)))
        test_rmse.append(np.sqrt(mean_squared_error(y_test,  test_sum / i)))

    rounds = np.arange(1, n_trees + 1)

    best_idx = int(np.argmin(test_rmse))
    best_test = test_rmse[best_idx]
    # best_train = train_rmse[best_÷idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.5)
    ax.plot(rounds, test_rmse,  label='Test RMSE',  linewidth=1.5)

    ax.fill_between(rounds, train_rmse, test_rmse,
                    alpha=0.15, color='red', label='Overfit gap')

    ax.axvline(x=best_idx + 1, color='green', linestyle='--',
               alpha=0.6, label=f'Best n_estimators ({best_idx + 1})')
    ax.scatter(best_idx + 1, best_test, color='green', zorder=5, s=80,
               edgecolors='black', linewidths=1.2)

    ax.annotate(f'Best Test RMSE: {best_test:.4f}',
                xy=(best_idx + 1, best_test),
                xytext=(best_idx + 5, best_test + 0.3),
                fontsize=9, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    ax.set_ylim(0, 7)
    ax.set_xlim(left=0, right=n_trees + 10)
    ax.set_title(f'{model_name} — Overfitting Diagnosis',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('RMSE (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(f"{model_name}_learning_curve", PLT_SAVE_DIR)
    plt.show()

    return best_idx + 1
