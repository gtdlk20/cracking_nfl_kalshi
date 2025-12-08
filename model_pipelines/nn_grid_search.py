import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from processors.batch_generator import DataPrepper
from tensorflow import keras
from tensorflow.keras import layers
from itertools import product

def GRU_generator():
    gru_size = [16,64]
    lookback = [12,48]
    dense = [16,64]
    dropout = [0.1,0.3]
    sentiment = ['oai','vader']
    param_strings = ['gru_size','dropout','dense_units','lookback','sentiment']
    model_params = list(product(gru_size, dropout, dense, lookback, sentiment))

    for i, (gru_size, dropout, dense_units, lookback, sentiment) in enumerate(model_params):
        # 1) Prep data for this config (adjust args to DataPrepper as needed)
        dp = DataPrepper(oai_vader_only=sentiment, lookback=lookback)

        # 2) Build model for this set of hyperparams
        model = keras.Sequential([
            keras.Input(shape=dp.X_train.shape[1:]),        # (timesteps, features)
            layers.GRU(gru_size, return_sequences=False, dropout=dropout),
            layers.Dense(dense_units, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )

        progress_string = f"Trained GRU model {i+1}/{len(model_params)}"
        yield model, dp, progress_string, (gru_size, dropout, dense_units, lookback, sentiment), param_strings

def CNN_generator():

    c2_filters = [8,32]
    c2_kernel = [3,7]
    dense = [16,64]
    dropout = [0.1,0.3]
    sentiment = ['oai','vader']
    lookback = [12,48]
    param_strings = ['c2_filters','c2_kernel','dense_units','dropout','lookback','sentiment']
    model_params = list(product(c2_filters, c2_kernel, dense, dropout, lookback, sentiment))

    for i, (c2_filters, c2_kernel, dense, dropout, lookback, sentiment) in enumerate(model_params):
        progress_string = f"Trained CNN model {i+1}/{len(model_params)}"

        dp = DataPrepper(oai_vader_only=sentiment, lookback=lookback)
        model = keras.Sequential([
            keras.Input(shape=dp.X_train.shape[1:]),

            # 1st conv block
            layers.Conv1D(filters=8, kernel_size=3, padding="causal", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            # 2nd conv block
            layers.Conv1D(filters=c2_filters, kernel_size=c2_kernel, padding="causal", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout),
            # Flatten + dense head
            layers.Flatten(),
            layers.Dense(dense, activation="relu"),
            layers.Dense(1)   # single value output (regression)
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        progress_string = f"Trained CNN model {i+1}/{len(model_params)}"
        yield model, dp, progress_string, (c2_filters, c2_kernel, dense, dropout, lookback, sentiment), param_strings


def model_generator():
    # generators = [CNN_generator(),GRU_generator()]
    generators = [GRU_generator()]
    for generator in generators:
        for model, dp, progress_string, params, param_strings in generator:
            yield model, dp, progress_string, params, param_strings