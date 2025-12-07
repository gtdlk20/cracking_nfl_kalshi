import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from batch_generator import DataPrepper

def main():
    # Load and preprocess data
    dp = DataPrepper()

    # define model
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(dp.X_train.shape[1], 1)))
    model.add(GRU(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

if __name__ == "__main__":
    main()