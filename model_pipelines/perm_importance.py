import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from processors.batch_generator import DataPrepper
from tensorflow import keras
from tensorflow.keras import layers
import model_pipelines.nn_grid_search as nn_grid_search
import matplotlib.pyplot as plt
import pickle

def compute_permutation_importance(model=None, dp=None):
    perm_importance = {feature: None for feature in dp.feature_cols}
    baseline = model.evaluate(dp.X_val_full, dp.y_val_full, batch_size=32, verbose=0)

    rng = np.random.default_rng(42) 
    for j,feature in enumerate(dp.feature_cols):
        deltas = []

        for _ in range(5):  # number of permutations
            X_perm = dp.X_val_full.copy()
            N, T, F = X_perm.shape
            vals = X_perm[:, :, j].reshape(-1)
            shuffled_vals = rng.permutation(vals)
            X_perm[:, :, j] = shuffled_vals.reshape(N, T)
            score = model.evaluate(X_perm, dp.y_val_full, batch_size=32, verbose=0)
            delta = score - baseline
            deltas.append(delta)

        perm_importance[feature] = np.mean(deltas)

    perm_df = pd.DataFrame(perm_importance, index=['importance']).T.sort_values(by='importance', ascending=False)
    perm_df.to_csv('store_models/perm_importance.csv')  
    return perm_df

def main():
    with open("store_models/best_dp.pkl", "rb") as f:
        dp = pickle.load(f)
    model = keras.models.load_model('store_models/best_model.keras')

    compute_permutation_importance(model, dp)

if __name__ == "__main__":
    main()