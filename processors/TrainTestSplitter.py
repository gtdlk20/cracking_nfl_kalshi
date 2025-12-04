from typing import Tuple
import pandas as pd


class TrainTestSplitter():
    """
    Processor for splitting data into training / validation / testing sets by ISO week.

    - Takes the full DataFrame at init and keeps train/val/test as attributes:
        self.train_df, self.val_df, self.test_df
    - Uses `threshold` to decide how many of the earliest weeks go to the (initial) training block.
    - Produces a validation set whose number of weeks is equal to the test set where possible:
        the validation weeks are taken from the most recent weeks of the training block.
    """

    def __init__(self, df: pd.DataFrame, date_col: str, threshold: float = 0.8):
        """
        Args:
        df: Full input DataFrame (will be copied).
        date_col: Column name containing datetimes.
        threshold: Fraction (0 < threshold < 1) of weeks to allocate to training
                before carving out validation from the end of that training block.
        """
        if not (0 < threshold < 1):
            raise ValueError("threshold must be between 0 and 1 (exclusive)")

        if date_col not in df.columns:
            raise KeyError(f"Date column '{date_col}' not found in DataFrame")

        self.date_col = date_col
        self.threshold = threshold
        self.df = df.copy()
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # initialize outputs
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self._split_weeks()

    def _split_weeks(self):
        if self.df.empty:
            self.train_df = self.df.copy()
            self.val_df = self.df.copy().iloc[0:0]
            self.test_df = self.df.copy().iloc[0:0]
            return

        weeks = self.df[self.date_col].dt.to_period("W")
        unique_weeks = sorted(weeks.drop_duplicates().tolist())
        n_weeks = len(unique_weeks)

        if n_weeks == 0:
            self.train_df = self.df.iloc[0:0].copy()
            self.val_df = self.df.iloc[0:0].copy()
            self.test_df = self.df.iloc[0:0].copy()
            return

        if n_weeks == 1:
            # all data -> train, no val/test
            self.train_df = self.df.reset_index(drop=True)
            self.val_df = self.df.iloc[0:0].reset_index(drop=True)
            self.test_df = self.df.iloc[0:0].reset_index(drop=True)
            return

        cutoff_idx = int(n_weeks * self.threshold)
        # ensure at least one week in train and one in test
        if cutoff_idx <= 0:
            cutoff_idx = 1
        if cutoff_idx >= n_weeks:
            cutoff_idx = n_weeks - 1

        n_test_weeks = n_weeks - cutoff_idx
        # try to make validation weeks equal to test weeks by taking them from the end of the train block,
        # but leave at least one week in the final training set
        n_val_weeks = min(n_test_weeks, max(0, cutoff_idx - 1))

        train_block = unique_weeks[:cutoff_idx]
        val_weeks = train_block[-n_val_weeks:] if n_val_weeks > 0 else []
        train_weeks = train_block[: len(train_block) - n_val_weeks] if n_val_weeks > 0 else train_block
        test_weeks = unique_weeks[cutoff_idx:]

        self.train_df = self.df[weeks.isin(train_weeks)].reset_index(drop=True)
        self.val_df = self.df[weeks.isin(val_weeks)].reset_index(drop=True)
        self.test_df = self.df[weeks.isin(test_weeks)].reset_index(drop=True)

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns (train_df, val_df, test_df) as copies."""
        return self.train_df.copy(), self.val_df.copy(), self.test_df.copy()

    def get_train(self) -> pd.DataFrame:
        """Return a copy of the training DataFrame."""
        return self.train_df.copy()

    def get_validation(self) -> pd.DataFrame:
        """Return a copy of the validation DataFrame."""
        return self.val_df.copy()

    def get_test(self) -> pd.DataFrame:
        """Return a copy of the testing DataFrame."""
        return self.test_df.copy()

    def summary(self) -> str:
        """Returns a string summary of the train/val/test split."""
        n_total = len(self.df)
        n_train = len(self.train_df)
        n_val = len(self.val_df)
        n_test = len(self.test_df)

        summary_str = (
            f"Total samples: {n_total}\n"
            f"Training samples: {n_train} ({(n_train / n_total * 100):.2f}%)\n"
            f"Validation samples: {n_val} ({(n_val / n_total * 100):.2f}%)\n"
            f"Testing samples: {n_test} ({(n_test / n_total * 100):.2f}%)"
        )
        return summary_str