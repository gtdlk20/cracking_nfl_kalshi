"""a class which takes a gru model, inherits its fit and predict methods, and wraps them in a pipeline step leveraging the kalshi_processor as a preprocessor step."""
from sklearn.pipeline import Pipeline
from processors.DataProcessor import DataProcessor
from models.GRU import GRUModel

class GRUPipeline():
    """Pipeline that integrates Kalshi data processing with a GRU model."""

    def __init__(self, time_res: str = 'day', datetime_cols=None, ffill_cols=None,
                 gru_units=64, dense_units=32):
        # Initialize the data processor
        self.processor = DataProcessor(time_res=time_res,
                                             datetime_cols=datetime_cols,
                                             ffill_cols=ffill_cols)
        
        # Placeholder input shape; will be set after processing data
        input_shape = (None, None)  
        
        # Initialize the GRU model
        self.model = GRUModel(input_shape=input_shape,
                              gru_units=gru_units,
                              dense_units=dense_units)
        
        # Create the pipeline
        self.pipeline = Pipeline(steps=[
            ('data_processor', self.processor),
            ('gru_model', self.model)
        ])

    def fit(self, X, y, epochs=10, batch_size=32):
        """Fit the pipeline on the training data."""
        # Process the data
        X_processed = self.processor.process(X)
        
        # Update input shape for the model
        self.model.model.build(input_shape=X_processed.shape)
        
        # Train the GRU model
        self.model.train(X_processed, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """Make predictions using the pipeline."""
        # Process the data
        X_processed = self.processor.process(X)
        
        # Make predictions with the GRU model
        return self.model.predict(X_processed)