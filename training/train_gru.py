from model_pipelines.GRU_pipeline import GRUPipeline
from utils.constants import KALSHI_FEATURE_COLS

__all__ = ['GRUPipeline']

"""A script which trains the GRU pipeline on Kalshi data."""

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load your Kalshi data
    data = pd.read_csv('data/nfl_historic_candlestick_minute.pkl')

    # Define features and target
    feature_cols = KALSHI_FEATURE_COLS  
    target_col = 'price_close'  

    X = data[feature_cols]
    y = data[target_col]

    # Split the data
    """This needs to be a custom train-test split, discerning on the week of the nfl season, to avoid data leakage."""
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the GRU pipeline
    gru_pipeline = GRUPipeline(time_res='minute',
                               datetime_cols=['datetime'],  # Update with actual datetime columns
                               ffill_cols=feature_cols,
                               gru_units=64,
                               dense_units=32)

    # Train the model
    gru_pipeline.fit(X_train, y_train, epochs=20, batch_size=64)

    # Validate the model
    predictions = gru_pipeline.predict(X_val)

    # Evaluate performance (e.g., MSE)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_val, predictions)
    print(f'Validation MSE: {mse}')