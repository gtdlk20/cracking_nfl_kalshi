import pandas as pd

class DataLoader:
    """A utility class for loading data from various sources."""

    def __init__(self, path: str):
        self.path = path
        if self.path.endswith('.csv'):
            self.load_data = self._load_data_csv
        elif self.path.endswith('.pkl'):
            self.load_data = self._load_data_pkl
        else:
            raise ValueError("Unsupported file format. Please use .csv or .pkl files.")
        

    def _load_data_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file into a DataFrame."""
        return pd.read_csv(file_path)

    def _load_data_pkl(self, file_path: str) -> pd.DataFrame:
        """Load data from a pickle file into a DataFrame."""
        return pd.read_pickle(file_path)