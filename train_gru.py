import os
import pandas as pd
from keras.models import load_model
from model_pipelines.GRU_pipeline import GRUPipeline
from utils.constants import KALSHI_FEATURE_COLS

if __name__ == "__main__":
    if not os.path.exists('store_models/gru_model.keras'):
        os.makedirs('store_models', exist_ok=True)
        # Load data and ensure datetime / target types are correct, also sort by time
        data = pd.read_csv('data/kalshi_reddit_sentiment_combined.csv')[KALSHI_FEATURE_COLS + ['end_period_ts', 'price_close']]
        data['end_period_ts'] = pd.to_datetime(data['end_period_ts'])
        data['price_close'] = pd.to_numeric(data['price_close'], errors='coerce')
        data = data.dropna(subset=['end_period_ts', 'price_close']).sort_values('end_period_ts').reset_index(drop=True)

        # Define features and target
        feature_cols = KALSHI_FEATURE_COLS  
    
        # Initialize the GRU pipeline
        gru_pipeline = GRUPipeline(
                                    data,
                                    datetime_cols=['end_period_ts'],
                                    ffill_cols=feature_cols,
                                    gru_units=64,
                                    dense_units=32
                                )


        # Train the model with batch_size=1 to avoid potential reshape mismatches for small datasets
        gru_pipeline.fit(epochs=4, batch_size=1)
    else:
        print("Model already exists. Loading the existing model for evaluation.")
        # Load data and ensure datetime / target types are correct, also sort by time
        data = pd.read_csv('data/kalshi_reddit_sentiment_combined.csv')[KALSHI_FEATURE_COLS + ['end_period_ts', 'price_close']]
        data['end_period_ts'] = pd.to_datetime(data['end_period_ts'])
        data['price_close'] = pd.to_numeric(data['price_close'], errors='coerce')
        data = data.dropna(subset=['end_period_ts', 'price_close']).sort_values('end_period_ts').reset_index(drop=True)

        # Define features and target
        feature_cols = KALSHI_FEATURE_COLS  
    
        # Initialize the GRU pipeline
        gru_pipeline = GRUPipeline(
                                    data,
                                    datetime_cols=['end_period_ts'],
                                    ffill_cols=feature_cols,
                                    gru_units=64,
                                    dense_units=32
                                )
        # Load the trained model
        gru_pipeline.model = gru_pipeline.model.load('store_models/gru_model.keras')

    # Validate the model
    y_true_val, y_pred_val, mse_val = gru_pipeline.evaluate(gru_pipeline.predict_on_val)
    print(f'Validation MSE: {mse_val}')
    # Evaluate performance on test data
    y_true_test, y_pred_test, mse_test = gru_pipeline.evaluate(gru_pipeline.predict_on_test)
    print(f'Test MSE: {mse_test}')
