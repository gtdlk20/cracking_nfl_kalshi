import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FFillImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, axis=0, limit=None):
        self.columns = columns
        self.axis = axis
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns:
            for col in self.columns:
                X_copy[col] = X_copy[col].ffill(axis=self.axis, limit=self.limit)
        else:
            X_copy = X_copy.ffill(axis=self.axis, limit=self.limit)
        return X_copy