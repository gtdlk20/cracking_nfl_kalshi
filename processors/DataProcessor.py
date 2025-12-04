import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers.DateTimeTransformer import DateTimeTransformer
from transformers.FFillImputer import FFillImputer
from utils.constants import KALSHI_DATETIME_COLS

class DataProcessor():
    """Processor for loading and processing Kalshi NFL candlestick data."""

    def __init__(self, datetime_cols=KALSHI_DATETIME_COLS, ffill_cols=None):

        self.datetime_cols = datetime_cols
    
        self.pipeline = Pipeline(steps=[
            ('datetime_transformer', DateTimeTransformer(self.datetime_cols)),
            ("ffill_imputer", FFillImputer(columns=ffill_cols, axis=0, limit=None)),
            ("standard_scaler", StandardScaler())
        ])

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame using the defined pipeline."""
        df[df.columns] = self.pipeline.fit_transform(df)
        return df

