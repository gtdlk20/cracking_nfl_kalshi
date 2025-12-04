"""a class which takes a gru model, inherits its fit and predict methods, and wraps them in a pipeline step leveraging the kalshi_processor as a preprocessor step."""
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from processors.DataProcessor import DataProcessor
from processors.TrainTestSplitter import TrainTestSplitter
from models.GRU import GRUModel
from utils.constants import KALSHI_FEATURE_COLS, KALSHI_DATETIME_COLS

class GRUPipeline():
    """Pipeline that integrates Kalshi data processing with a GRU model."""

    def __init__(self, data, datetime_cols=None, ffill_cols=None,
                 gru_units=64, dense_units=32):
        # Initialize the data processor
        self.processor = DataProcessor(
                            datetime_cols=datetime_cols,
                            ffill_cols=ffill_cols
                        )
        self.data = TrainTestSplitter(data, date_col=KALSHI_DATETIME_COLS[0])
        self.train_data = self.data.get_train()
        self.val_data = self.data.get_validation()
        self.test_data = self.data.get_test()
        
        # Initialize the GRU model
        self.model = GRUModel(input_shape=(None, len(KALSHI_FEATURE_COLS)),
                              gru_units=gru_units,
                              dense_units=dense_units)

    def fit(self,epochs=10, batch_size=32):
        """Fit the pipeline on the training data."""
        if self.train_data.empty:
            raise ValueError("Training data is empty. Cannot fit the model.")
        # Process the data
        X = self.train_data.drop(columns=['price_close'])
        X_processed = self.processor.process(X)
        print(type(X_processed))
        y = self.train_data['price_close']  
        # Update input shape for the model
        self.model.model.build(input_shape=X_processed.shape)
        
        # Train the GRU model
        self.model.train(X_processed, y, epochs=epochs, batch_size=batch_size)
        # Save the trained model
        self.model.save('store_models/gru_model.keras')
        print("Model training complete and saved.")

    def predict_on_test(self):
        """Make predictions using the pipeline."""
        # Process the data
        X = self.test_data.drop(columns=['price_close'])
        X_processed = self.processor.process(X)
        return self.model.predict(X_processed)

    def predict_on_val(self):
        """Make predictions using the pipeline."""
        # Process the data
        X = self.val_data.drop(columns=['price_close'])
        X_processed = self.processor.process(X) 
        # Make predictions with the GRU model
        return self.model.predict(X_processed)
    
    def predict(self, X):
        """Make predictions using the pipeline. Assumes data is pre-split."""
        # Process the data
        if 'price_close' in X.columns:
            X = X.drop(columns=['price_close'])

        X_processed = self.processor.process(X)
        # Make predictions with the GRU model
        return self.model.predict(X_processed)
    
    def evaluate(self, predict, X=None):
        """Evaluate the model on the test data."""
        y_true = self.test_data['price_close']
        y_pred = predict(self.test_data) if X is None else predict(X)
        mse = mean_squared_error(y_true, y_pred)
        return y_true, y_pred, mse