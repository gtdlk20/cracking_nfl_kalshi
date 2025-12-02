import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers
import pandas as pd

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
    def prepare_inputs(data,
                       sample_id_col=None,
                       time_col='timestamp',
                       maxlen=None,
                       padding="pre",
                       value=0.0,
                       features=None,
                       sort_time=True):
        """
        Build model inputs from pandas DataFrame(s).

        Accepts either:
         - a single pandas.DataFrame containing one sample (sequence) OR multiple samples
           (in which case `sample_id_col` must be provided to group rows into samples),
         - OR an iterable (list) of pandas.DataFrame objects (one per sample).

        Expected columns (defaults / common synonyms are supported):
            timestamp (or time / datetime / ts)
            trade_volume (or volume)
            close price (close / close_price)
            high price (high / high_price)
            low price (low / low_price)
            popularity-weighted sentiment (or pop_sentiment / popularity_weighted_sentiment)
            vader positive component (vader_pos / vader_positive)
            vader neutral component (vader_neu / vader_neutral)
            vader negative component (vader_neg / vader_negative)

        Args:
            data: pd.DataFrame, or list of pd.DataFrame samples.
            sample_id_col: column name used to split a single DataFrame into samples (optional).
            time_col: name of timestamp column to sort by.
            maxlen: target sequence length (if None uses longest sequence).
            padding: 'pre' or 'post' placement for padding.
            value: padding value (will be masked by model).
            features: optional list of column names (or synonyms) to use and their order. If None,
                      uses the default set described above in the order:
                      [trade_volume, close, high, low, pop_sentiment, vader_pos, vader_neu, vader_neg]
            sort_time: whether to sort each sample by time_col.

        Returns:
            np.ndarray shaped (batch, maxlen, n_features)
        """

        # helper to find a column from possible synonyms
        def _find_col(df_cols, options):
            for o in options:
                if o in df_cols:
                    return o
            return None

        # default feature sets / synonyms
        if features is None:
            features_to_find = [
                ['trade_volume', 'volume', 'trade volume'],
                ['close price', 'close_price', 'close', 'price_close'],
                ['high price', 'high_price', 'high'],
                ['low price', 'low_price', 'low'],
                ['popularity-weighted sentiment', 'popularity_weighted_sentiment', 'pop_weighted_sentiment', 'pop_sentiment', 'popularity_sentiment'],
                ['vader_positive', 'vader_pos', 'vader positive', 'vader_positive_component'],
                ['vader_neutral', 'vader_neu', 'vader neutral', 'vader_neutral_component'],
                ['vader_negative', 'vader_neg', 'vader negative', 'vader_negative_component']
            ]
        else:
            # user-provided: allow a list of column names (strings)
            features_to_find = [[f] for f in features]

        # normalize input to list of dataframes (samples)
        samples = []
        if isinstance(data, pd.DataFrame):
            if sample_id_col is None:
                # treat entire DataFrame as single sample
                samples = [data.copy()]
            else:
                # group by sample id column
                groups = data.groupby(sample_id_col)
                samples = [g.copy() for _, g in groups]
        else:
            # assume iterable of dataframes
            samples = [d.copy() for d in data]

        df_cols = samples[0].columns if len(samples) > 0 else []
        # resolve actual column names for features
        resolved_cols = []
        for opts in features_to_find:
            col = _find_col(df_cols, opts)
            resolved_cols.append(col)

        # Verify all required columns exist in at least the first sample's columns.
        missing = [opts for opts, col in zip(features_to_find, resolved_cols) if col is None]
        if any(col is None for col in resolved_cols):
            missing_readable = [opts[0] for opts in missing]
            raise ValueError(f"Could not find required columns. Expected (synonyms accepted): {missing_readable}")

        # optional timestamp column check
        tc = _find_col(df_cols, [time_col, 'timestamp', 'time', 'datetime', 'ts'])
        if tc is None and sort_time:
            # no timestamp available; sorting disabled
            sort_time = False
        else:
            time_col = tc if tc is not None else time_col

        sequences = []
        for df in samples:
            # ensure we have the resolved columns for this df (some samples may have differing names)
            df_cols_local = df.columns
            resolved_local = []
            for opts in features_to_find:
                col_local = _find_col(df_cols_local, opts)
                if col_local is None:
                    raise ValueError(f"Sample missing required column. Expected one of {opts}.")
                resolved_local.append(col_local)

            # optionally sort
            if sort_time:
                df = df.sort_values(by=time_col)

            # extract features in order
            arr = df[resolved_local].to_numpy(dtype='float32')  # shape (timesteps, n_features)
            sequences.append(arr)

        # determine maxlen
        lengths = [s.shape[0] for s in sequences]
        if maxlen is None:
            maxlen = max(lengths) if lengths else 0

        n_features = sequences[0].shape[1] if sequences else 0
        batch = len(sequences)
        out = np.full((batch, maxlen, n_features), fill_value=value, dtype='float32')

        for i, seq in enumerate(sequences):
            L = seq.shape[0]
            if L == 0:
                continue
            if L > maxlen:
                # truncate according to padding/truncating strategy (match pad_sequences behavior)
                if padding == 'pre':
                    seq = seq[-maxlen:, :]
                    L = maxlen
                else:
                    seq = seq[:maxlen, :]
                    L = maxlen

            if padding == 'pre':
                out[i, maxlen - L: maxlen, :] = seq
            else:
                out[i, :L, :] = seq

        return out
    def __init__(self, input_shape, gru_units=64, dense_units=32):
        self.model = models.Sequential()
        self.model.add(layers.GRU(gru_units, input_shape=input_shape))
        self.model.add(layers.Dense(dense_units, activation='relu'))
        self.model.add(layers.Dense(1))  # Assuming a regression task

        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=10, batch_size=32):
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