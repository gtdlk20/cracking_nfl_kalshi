import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers
from utils.constants import KALSHI_FEATURE_COLS
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class GRUModel:
    @staticmethod
    def build_model(sequence_length=None,
                    feature_dim=2,
                    gru_units=(128, 64),
                    dropout=0.2,
                    dense_units=(64, 32),
                    lr=1e-3):
        """
        Builds a robust GRU-based regression model for minute-by-minute price + sentiment
        time series. Uses masking, 2-layer bidirectional GRU stack, temporal attention,
        and a small dense head. Returns a compiled Keras Model.

        Args:
            sequence_length: optional fixed sequence length (None allows variable length).
            feature_dim: number of features per timestep (e.g., [price_delta, vader_score]).
            gru_units: tuple for two GRU layers.
            dropout: dropout rate between recurrent layers.
            dense_units: tuple for dense head.
            lr: learning rate for Adam.

        Returns:
            tf.keras.Model compiled for regression (MSE).
        """
        inp = layers.Input(shape=(sequence_length, feature_dim), name="time_series_input")
        x = layers.Masking(mask_value=0.0)(inp) 

        # First bidirectional GRU (return sequences to preserve temporal info)
        x = layers.Bidirectional(
            layers.GRU(gru_units[0], return_sequences=True, dropout=dropout, recurrent_dropout=0.0)
        )(x)

        # Second bidirectional GRU
        x = layers.Bidirectional(
            layers.GRU(gru_units[1], return_sequences=True, dropout=dropout, recurrent_dropout=0.0)
        )(x)

        # Temporal attention mechanism (learned weighting over timesteps)
        att_scores = layers.Dense(1, use_bias=False)(x)              # (batch, time, 1)
        att_weights = layers.Softmax(axis=1, name="att_weights")(att_scores)  # normalize over time
        context = layers.Dot(axes=1)([att_weights, x])               # (batch, 1, features)
        context = layers.Flatten()(context)                          # (batch, features)

        # Dense head with normalization and dropout
        x = layers.LayerNormalization()(context)
        x = layers.Dense(dense_units[0], activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(dense_units[1], activation="relu")(x)

        out = layers.Dense(1, activation="linear", name="price_close")(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
        return model

    @staticmethod
    def prepare_inputs(df: pd.DataFrame,
                       feature_cols=None,
                       maxlen=None,
                       padding="pre",
                       value=0.0):
        """
        Build a (batch, time, features) array from a DataFrame where each cell in
        `feature_cols` is a sequence (list/array) of timesteps.
        Solely pads/truncates sequences to `maxlen`. Feature preprocessing handled elsewhere.

        Args:
            df: DataFrame with one sample per row and sequence-valued columns.
            feature_cols: list of column names to use; defaults to KALSHI_FEATURE_COLS.
            maxlen: target sequence length (if None uses longest sequence found).
            padding: 'pre' or 'post' for pad_sequences.
            value: padding value (will be masked).

        Returns:
            np.ndarray shaped (batch, maxlen, n_features).
        """
        if feature_cols is None:
            feature_cols = list(KALSHI_FEATURE_COLS)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"The following feature columns are missing from df: {missing}")

        # determine maxlen if not provided
        if maxlen is None:
            maxlen = 0
            for col in feature_cols:
                for cell in df[col].values:
                    if pd.isna(cell):
                        continue
                    arr = np.asarray(cell)
                    length = 1 if arr.ndim == 0 else arr.shape[0]
                    if length > maxlen:
                        maxlen = length

        per_feature_pads = []
        for col in feature_cols:
            seqs = []
            for cell in df[col].values:
                if pd.isna(cell):
                    seqs.append(np.array([], dtype="float32"))
                else:
                    arr = np.asarray(cell, dtype="float32")
                    if arr.ndim == 0:
                        arr = arr.reshape(1)
                    seqs.append(arr)
            per_feature_pads.append(
                pad_sequences(seqs, maxlen=maxlen, dtype="float32", padding=padding, truncating=padding, value=value)
            )

        # stack features into (batch, time, features)
        combined = np.stack(per_feature_pads, axis=-1)
        return combined

    def __init__(self, input_shape, gru_units=64, dense_units=32):
        self.model = self.build_model(
            sequence_length=input_shape[1],
            feature_dim=input_shape[2],
            gru_units=(gru_units, gru_units // 2),
            dense_units=(dense_units, dense_units // 2)
        )

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        X_train = self.prepare_inputs(X_train)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        model = GRUModel((None, None))  # Placeholder shape
        model.model = models.load_model(filepath)
        return model