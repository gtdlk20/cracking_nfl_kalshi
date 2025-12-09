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
from assess_model import plot_model
import argparse

def retrain_model():
    with open("store_models/cv_results.pkl", "rb") as f:
        cv_results = pickle.load(f)
    model = keras.models.load_model('store_models/best_model.keras')

    params = sorted(cv_results, key=lambda x: x['val_loss'])[0]
    dp_sentiment = params['sentiment']
    dp_lookback = params['lookback']
    dp = DataPrepper(oai_vader_only=dp_sentiment, lookback=dp_lookback)

    history = model.fit(dp.X_train, dp.y_train, validation_data=(dp.X_val_change, dp.y_val_change),epochs=25, batch_size=32)

    plot_model(model, dp, history)
    with open("store_models/best_dp.pkl", "wb") as f:
        pickle.dump(dp, f)
    with open("store_models/best_history.pkl", "wb") as f:
        pickle.dump(history, f)

def main():
    args = parse_args()
    gen = nn_grid_search.model_generator()

    if args.rt:
        retrain_model()
        return

    best_model = None
    best_params = None
    best_history = None
    best_dp = None
    best_val_loss = float('inf')

    cv_results = []
    for model, dp, progress_string, params, param_strings in gen:

        history = model.fit(dp.X_train, dp.y_train, validation_data=(dp.X_val_change, dp.y_val_change),epochs=10, batch_size=32, verbose=0)
        best_model_val_loss = history.history['val_loss'][-1]

        key = {param_name: param_value for param_name, param_value in zip(param_strings, params)} 
        hist_key = key.copy()
        hist_key['val_loss'] = best_model_val_loss
        if best_model_val_loss < best_val_loss:
            best_val_loss = best_model_val_loss
            best_model = model
            best_params = key
            best_history = history
            best_dp = dp

        param_combo_string = ', '.join([f"{k}={v}" for k, v in key.items()])
        cv_results.append(hist_key)

        print(f"{progress_string} \twith {param_combo_string}, \tval_loss={best_model_val_loss:.3f}")

    # save best values
    best_model.save('store_models/best_model.keras')
    with open("store_models/best_dp.pkl", "wb") as f:
        pickle.dump(best_dp, f)
    with open("store_models/best_history.pkl", "wb") as f:
        pickle.dump(best_history, f)
    with open("store_models/cv_results.pkl", "wb") as f:
        pickle.dump(cv_results, f)
    plot_model(best_model, best_dp, best_history)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain model with more epochs on best hyperparameters."
    )
    parser.add_argument(
        "--rt",
        action="store_true",
        help="retrain model with best params",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
