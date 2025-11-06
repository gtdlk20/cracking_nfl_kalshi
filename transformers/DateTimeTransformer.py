import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert timestamp columns to datetime objects."""

    def __init__(self, timestamp_columns):
        self.timestamp_columns = timestamp_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.timestamp_columns:
            X_transformed[col] = pd.to_datetime(X_transformed[col], unit='s')
        return X_transformed