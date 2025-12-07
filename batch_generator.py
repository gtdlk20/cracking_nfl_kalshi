import pandas as pd
import numpy as np
from utils.constants import KALSHI_FEATURE_COLS, KALSHI_FEATURE_COLS_OAI, KALSHI_FEATURE_COLS_VADER
from sklearn.preprocessing import StandardScaler

class DataPrepper():
    def __init__(self, path='data/kalshi_reddit_sentiment_combined_15min.pkl', target_col='price_close_next', oai_vader_only=None):
        self.path = path
        self.target_col = target_col
        self.feature_cols = KALSHI_FEATURE_COLS 
        if oai_vader_only == 'oai':
            self.feature_cols = KALSHI_FEATURE_COLS_OAI
        elif oai_vader_only == 'vader':
            self.feature_cols = KALSHI_FEATURE_COLS_VADER
        self.data = pd.read_pickle(path).reset_index()[self.feature_cols + ['end_period_ts','team','opp', target_col]]
        self.train_df, self.val_df, self.test_df = self.train_test_splitter()

        # get unbatched data
        self.X_train, self.y_train = self._recombine_batches(ttv='train')
        self.X_val, self.y_val = self._recombine_batches(ttv='val')
        self.X_test, self.y_test = self._recombine_batches(ttv='test')

    def df_to_3d(self, df, lookback):
        """
        df: pandas DataFrame, shape (T, F)
        lookback: int
        returns: np.ndarray, shape (T - lookback + 1, lookback, F)
        """
        values = df.to_numpy()
        T, F = values.shape
        L = lookback

        X = np.empty((T - L + 1, L, F), dtype=values.dtype)
        for i in range(T - L + 1):
            X[i] = values[i:i+L]
        return X

    def train_test_splitter(self):
        data = self.data.sort_values('end_period_ts',ascending=True).reset_index(drop=True)
        data['end_period_ts'] = data['end_period_ts'].dt.tz_localize('UTC').dt.tz_convert('US/Central')
        data['dow'] = data['end_period_ts'].dt.day_of_week

        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for team in data['team'].unique():
            team_data = data[data['team']==team].copy()
            team_opp_map = {opp: i+1 for i,opp in enumerate(team_data['opp'].unique())}
            team_data['week'] = team_data['opp'].map(team_opp_map)

            # check that a week's last data in on sunday (6)
            week_end_check = team_data.groupby(['week','team','opp'])['dow'].max()
            team_data.set_index(['week','team','opp'], inplace=True)
            team_data = team_data.loc[week_end_check[week_end_check==6].index].reset_index()

            # check that a team has at least 3 weeks of data
            team_week_count = team_data[['team','opp']].drop_duplicates().groupby('team').count()
            valid_teams = team_week_count[team_week_count['opp']>=3].index.tolist()
            team_data = team_data.set_index('team').loc[valid_teams].reset_index()

            # for test use last week, val use week before last, train use all others
            weeks = team_data['week'].unique()
            test_weeks = [weeks[-1]]
            val_weeks = [weeks[-2]]
            train_weeks = weeks[:-2]

            # add to respective dfs
            train_df = pd.concat([train_df, team_data[team_data['week'].isin(train_weeks)]])
            val_df = pd.concat([val_df, team_data[team_data['week'].isin(val_weeks)]])
            test_df = pd.concat([test_df, team_data[team_data['week'].isin(test_weeks)]])

        # scaling
        scaler = StandardScaler()
        # scale all feature cols except price_close, price_high, price_low
        scale_cols = [c for c in self.feature_cols if c not in ('price_close', 'price_high', 'price_low')]
        train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])
        
        return train_df, val_df, test_df


    def batch_generator(self, lookback=10, batch_size=32, ttv='train'):    
        if ttv=='train':
            data = self.train_df
        elif ttv=='val':
            data = self.val_df
        else:
            data = self.test_df
        data = data.sort_values('end_period_ts',ascending=True).reset_index(drop=True)

        for team in data['team'].unique():
            team_data = data[data['team']==team].copy()
            weeks = team_data['week'].unique()

            # for each week, create 3d array and yield batches
            for weekn in weeks:
                week_data = team_data[team_data['week'] == weekn].copy()
                week_data = week_data[self.feature_cols+['price_close_next']]
                if len(week_data) < lookback*2:
                    continue
                arr = self.df_to_3d(week_data, lookback=lookback)
                week_data = week_data.iloc[-len(arr):].reset_index(drop=True)
                flat_idxs = week_data[week_data['price_close_next'] == week_data['price_close']].sample(frac=0.2, random_state=42).index
                change_idx = week_data[week_data['price_close_next'] != week_data['price_close']].index
                final_idxs = list(flat_idxs.union(change_idx).sort_values())
                # print(len(week_data), len(final_idxs))
                for start_idx in range(0, len(final_idxs), batch_size):
                    end_idx = start_idx + batch_size
                    yield arr[final_idxs[start_idx:end_idx], :, :-1], arr[final_idxs[start_idx:end_idx], -1, -1]

    def _recombine_batches(self, lookback=10, batch_size=32, ttv='train'):
        """
        Collect all batches produced by batch_generator and stack them into full arrays.
        Returns: X, y where
            X.shape == (N, lookback, n_features)
            y.shape == (N, lookback)
        """
        X_batches = []
        y_batches = []
        for Xb, yb in self.batch_generator(lookback=lookback, batch_size=batch_size, ttv=ttv):
            X_batches.append(Xb)
            y_batches.append(yb)

        X = np.concatenate(X_batches, axis=0)
        y = np.concatenate(y_batches, axis=0)

        return X, y
