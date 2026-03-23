# def plot_learning_curve(evals_result, model_name="XGBoost"):
#     """
#     Plot train vs test RMSE learning curve for a single model
#     to diagnose overfitting.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     train_rmse = evals_result['train']['rmse']
#     test_rmse = evals_result['test']['rmse']
#     rounds = np.arange(1, len(train_rmse) + 1)

#     # Final values
#     train_final = train_rmse[-1]
#     test_final = test_rmse[-1]
#     gap = test_final - train_final

#     # Best test round
#     best_idx = int(np.argmin(test_rmse))
#     best_test = test_rmse[best_idx]

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 5))

#     ax.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.5)
#     ax.plot(rounds, test_rmse,  label='Test RMSE',  linewidth=1.5)

#     # Shade overfitting gap
#     ax.fill_between(rounds, train_rmse, test_rmse,
#                     alpha=0.15, color='red', label='Overfit gap')

#     # Mark best test point
#     ax.axvline(x=best_idx + 1, color='grey', linestyle='--',
#                alpha=0.6, label=f'Best test round ({best_idx + 1})')
#     ax.scatter(best_idx + 1, best_test, color='red', zorder=5, s=60)

#     # Annotate final gap on the right side
#     last_round = rounds[-1]
#     mid_y = (train_final + test_final) / 2

#     # Draw bracket line showing the gap
#     ax.annotate('', xy=(last_round + 1, train_final),
#                 xytext=(last_round + 1, test_final),
#                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

#     ax.text(last_round + 2, mid_y, f'Gap: {gap:.2f}%',
#             fontsize=9, color='red',
#             va='center')

#     # Fixed y-axis for cross-model comparison
#     ax.set_ylim(0, 7)
#     ax.set_xlim(left=0, right=last_round + 10)  # extra space for annotation

#     ax.set_title(f'{model_name} — Overfitting Diagnosis',
#                  fontsize=13, fontweight='bold')
#     ax.set_xlabel('Boosting Round')
#     ax.set_ylabel('RMSE (%)')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     plt.show()

def plot_learning_curve(evals_result, model_name="XGBoost"):
    """
    Plot train vs test RMSE learning curve for a single model
    to diagnose overfitting.
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
        f"  Best round: {best_idx + 1}  |  Train RMSE: {best_train:.4f}  |  Test RMSE: {best_test:.4f}  |  Gap: {gap:.4f}")
