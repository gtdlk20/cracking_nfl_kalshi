from tensorflow import keras
from tensorflow.keras import layers
import model_pipelines.nn_grid_search as nn_grid_search
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

def plot_model(model, dp, history):
    loss = model.evaluate(dp.X_test_full, dp.y_test_full, verbose=0)
    predictions = model.predict(dp.X_test_full, verbose=0).flatten()

    print(model.summary())
    print(f"Test Loss: {loss:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax0 = axes[0, 0]
    ax0.plot(history.history['loss'], label='Training Loss')
    ax0.plot(history.history['val_loss'], label='Validation Loss')
    ax0.legend()
    ax0.set_title('Train/Val Loss')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('MSE Loss')

    ax1 = axes[0, 1]
    ax1.scatter(dp.y_test_full, predictions, alpha=0.5)
    min_val = min(predictions.min(), dp.y_test_full.min())
    max_val = max(predictions.max(), dp.y_test_full.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")

    ax3 = axes[1, 0]
    preds_df = pd.DataFrame({'current_val': dp.X_test_full[:, 0, 0], 'actual_next': dp.y_test_full, 'predicted_next': predictions})
    preds_df['predicted_change'] = preds_df['predicted_next'] - preds_df['current_val']
    # preds_df[preds_df['predicted_change'] > 0.02]
    preds_df['predicted_change_abs'] = preds_df['predicted_change'].abs()
    preds_df['actual_change'] = preds_df['actual_next'] - preds_df['current_val']
    preds_df = preds_df.sort_values(by='predicted_change_abs', ascending=False).reset_index(drop=True)
    preds_df['profit'] = preds_df['actual_change'] * np.sign(preds_df['predicted_change'])
    preds_df['profit_sum'] = preds_df['profit'].cumsum()
    preds_df['cost_sum'] = preds_df['current_val'].cumsum()
    preds_df['roi'] = preds_df['profit_sum'] / preds_df['cost_sum'] * 100
    preds_df = preds_df[(preds_df['current_val'] > 0.02)&(preds_df['current_val'] < 0.98)]
    # 1) Cumulative profit overall
    ax3.plot(preds_df['profit_sum'])
    ax3.set_title('Cumulative Profit by Ordered Predictions')
    ax3.set_xlabel('Possible Bets Ordered by Predicted Absolute Change')
    ax3.set_ylabel('Cumulative profit')
    # Find max cumulative profit and corresponding pred
    idx_max = preds_df['profit_sum'].idxmax()
    max_profit = preds_df.loc[idx_max, 'profit_sum']
    pred_at_max = preds_df.loc[idx_max, 'predicted_change_abs']
    # Highlight the point
    ax3.scatter(idx_max, max_profit, color='red', zorder=3)
    # Add callout / annotation
    roi_max = max_profit/preds_df.loc[idx_max, 'cost_sum'] * 100
    ax3.annotate(
        f"Max profit: ${max_profit:.2f}\nROI: {int(roi_max)}%\nPredicted Abs Change: {pred_at_max:.2f}",
        xy=(idx_max, max_profit),
        xytext=(idx_max, max_profit * 0.6),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    idx_max = preds_df['roi'].idxmax()
    max_roi = preds_df.loc[idx_max, 'roi']
    pred_at_max = preds_df.loc[idx_max, 'predicted_change_abs']
    profit_max = preds_df.loc[idx_max, 'profit_sum']
    ax3.scatter(idx_max, profit_max, color='red', zorder=3)
    ax3.annotate(
        f"Max ROI: {max_roi:.2f}%\nProfit: ${profit_max:.2f}\nPredicted Abs Change: {pred_at_max:.2f}",
        xy=(idx_max, profit_max),
        xytext=(idx_max, max_profit * 0.6),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax4 = axes[1, 1]
    preds_df['profitable'] = (preds_df['profit'] > 0).astype(int)
    grouped = preds_df.groupby(preds_df['predicted_change_abs'].round(2))[['profit', 'current_val']].sum().sort_index(ascending=False)
    grouped['profitable_mean'] = preds_df.groupby(preds_df['predicted_change_abs'].round(2))['profitable'].mean()

    grouped['roi'] = grouped['profit'] / grouped['current_val'] * 100
    grouped['profit_sum'] = grouped['profit'].cumsum()
    grouped['cost_sum'] = grouped['current_val'].cumsum()
    # fig, ax1 = plt.subplots()

    # Plot ROI on the primary y-axis
    ax4.plot(grouped.index, grouped['roi'], label='ROI', color='#1F77B4')
    ax4.set_xlabel('Predicted Abs Change (Rounded)')
    ax4.set_ylabel('ROI (%)', color='#1F77B4')     
    ax4.tick_params(axis='y', labelcolor='#1F77B4')

    # Create a secondary y-axis for profit
    ax5 = ax4.twinx()
    ax5.plot(grouped.index, grouped['profitable_mean'] * 100, label='% Profitable Predictions', color='green', alpha=0.6)
    ax5.set_ylabel('% Profitable Predictions', color='green')
    ax5.tick_params(axis='y', labelcolor='green')

    # Add legends
    plt.title('ROI and Profit by Predicted Change')

    plt.tight_layout()
    fig.suptitle(f'Model Evaluation – MSE: {loss:.3f}', fontsize=14, y=1.02)

    path = f'store_models/model_eval.png'
    fig.savefig(path, dpi=300, bbox_inches="tight")

def main():
    with open("store_models/best_dp.pkl", "rb") as f:
        dp = pickle.load(f)
    with open("store_models/best_history.pkl", "rb") as f:
        history = pickle.load(f)
    model = keras.models.load_model('store_models/best_model.h5')

    plot_model(model, dp, history)


if __name__ == "__main__":
    main()