import pandas as pd
from sklearn.pipeline import Pipeline
from transformers.DateTimeTransformer import DateTimeTransformer
from transformers.FFillImputer import FFillImputer
from utils.constants import KALSHI_DATA_DAY, KALSHI_DATA_HOUR

class KalshiDataProcessor:
    """Processor for loading and processing Kalshi NFL candlestick data."""

    def __init__(self, time_res: str = 'day', datetime_cols=['end_period_ts'], ffill_cols=None):
        if time_res == 'day':
            self.data_path = KALSHI_DATA_DAY
        elif time_res == 'hour':
            self.data_path = KALSHI_DATA_HOUR
        else:
            raise ValueError("Unsupported time resolution. Please use 'day' or 'hour'.")
        self.datetime_cols = datetime_cols
    
        self.pipeline = Pipeline(steps=[
            ('datetime_transformer', DateTimeTransformer(self.datetime_cols)),
            ("ffill_imputer", FFillImputer(columns=ffill_cols, axis=0, limit=None))
        ])

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame using the defined pipeline."""
        return self.pipeline.fit_transform(df)

