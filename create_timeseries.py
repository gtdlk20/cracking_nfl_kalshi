import pandas as pd
import numpy as np
import argparse
from utils.constants import REDDIT_DATA_PATH
from utils.constants import KALSHI_DATA_HOUR_CSV
from utils.constants import REDDIT_SUBS, KALSHI_FEATURE_COLS

def process_sentiment(sub_name, time_scale='h'):
    path = f"{REDDIT_DATA_PATH}/{sub_name}_sentiment.csv"
    df = pd.read_csv(path)
    dummies = pd.get_dummies(df['sentiment'], prefix='oai').astype(int)
    df = pd.concat([df, dummies], axis=1)
    cols = ['created_utc', 'populairity', 'oai_negative', 'oai_neutral', 'oai_positive', 'vader_neg','vader_neu','vader_pos']
    df = df[cols]
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df.set_index('time')
    df.drop('created_utc', axis=1, inplace=True)
    sent_cols = ['oai_negative', 'oai_neutral', 'oai_positive', 'vader_neg', 'vader_neu', 'vader_pos']
    df['log_pop'] = np.log(df['populairity']+1)
    for col in sent_cols:
        df[col] = df[col] * df['log_pop']
    df.drop(['populairity', 'log_pop'], axis=1, inplace=True)
    df = df.resample(time_scale)[sent_cols].agg(['mean', 'std']).fillna(0)
    df.columns = [f"{col}_{stat}" for col, stat in df.columns]
    df.sort_index(inplace=True)
    return df

def load_subreddits(time_scale='h'):
    subs_df = pd.read_csv(REDDIT_SUBS)
    subs_df.dropna(inplace=True)
    subs_list = []
    for _,row in subs_df.iterrows():
        sub_sent_df = process_sentiment(row['Subreddit'], time_scale=time_scale)
        sub_sent_df['team'] = row['Team']
        sub_sent_df = sub_sent_df.round(2)
        subs_list.append(sub_sent_df)

    sub_sent_df = pd.concat(subs_list)
    return sub_sent_df

def get_kalshi_data(time_scale='h'):
    market_df = pd.read_csv(KALSHI_DATA_HOUR_CSV)
    market_df['team'] = market_df['market'].apply(lambda x: x.split('-')[2])
    market_df['event'] = market_df['market'].apply(lambda x: x.split('-')[1])
    market_df = market_df[['end_period_ts','price.close_dollars','price.high_dollars','price.low_dollars','team','event']].ffill()
    market_df['end_period_ts'] = pd.to_datetime(market_df['end_period_ts'], unit='s')
    market_df = market_df.set_index('end_period_ts')
    market_df = market_df[market_df.index >=pd.to_datetime('2025-08-28')]
    market_df['opp'] = [event[7:].replace(team,'') for event,team in zip(market_df['event'], market_df['team'])]
    market_df.rename(columns={'price.close_dollars':'price_close','price.high_dollars':'price_high','price.low_dollars':'price_low'}, inplace=True)

    # resample by minute, keeping data from each team/event combo
    # also add in back shifted price data
    df_list = []
    for event in market_df['event'].unique():
        event_df = market_df[market_df['event']==event]
        for team in event_df['team'].unique():
            event_df_team = event_df[event_df['team']==team]
            event_df_team = event_df_team.resample(time_scale).ffill().fillna(0)
            event_df_team['price_close_prev'] = event_df_team['price_close'].shift(1)
            event_df_team['price_high_prev'] = event_df_team['price_high'].shift(1)
            event_df_team['price_low_prev'] = event_df_team['price_low'].shift(1)
            # event_df_team = event_df_team.fillna(0)
            df_list.append(event_df_team)
    market_df = pd.concat(df_list)
    market_df = market_df.reset_index()
    return market_df

def main(time_scale='h'):
    sub_sent_df = load_subreddits(time_scale=time_scale)
    market_df = get_kalshi_data(time_scale=time_scale)

    df = market_df.merge(sub_sent_df, left_on=['end_period_ts','team'], right_on=['time','team'], how='left')
    df = df.merge(sub_sent_df, left_on=['end_period_ts','opp'], right_on=['time','team'], how='left', suffixes=('', '_opp'))
    df.dropna(inplace=True)
    df.set_index('end_period_ts', inplace=True)
    df[KALSHI_FEATURE_COLS] = df[KALSHI_FEATURE_COLS].astype(float)
    df.sort_index(inplace=True)
    df.to_csv('data/kalshi_reddit_sentiment_combined.csv', index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create time series data by combining Kalshi market data with Reddit sentiment data.")
    parser.add_argument('--time_scale', '-ts', type=str, default='h', help="Time scale for resampling data (e.g., 'h' for hourly, 'd' for daily).")
    args = parser.parse_args()
import pandas as pd
import numpy as np
import argparse
from utils.constants import REDDIT_DATA_PATH
from utils.constants import KALSHI_DATA_HOUR_CSV
from utils.constants import REDDIT_SUBS

def process_sentiment(sub_name, time_scale='h'):
    path = f"{REDDIT_DATA_PATH}/{sub_name}_sentiment.csv"
    df = pd.read_csv(path)
    dummies = pd.get_dummies(df['sentiment'], prefix='oai').astype(int)
    df = pd.concat([df, dummies], axis=1)
    cols = ['created_utc', 'populairity', 'oai_negative', 'oai_neutral', 'oai_positive', 'vader_neg','vader_neu','vader_pos']
    df = df[cols]
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df.set_index('time')
    df.drop('created_utc', axis=1, inplace=True)
    sent_cols = ['oai_negative', 'oai_neutral', 'oai_positive', 'vader_neg', 'vader_neu', 'vader_pos']
    df['log_pop'] = np.log(df['populairity']+1)
    for col in sent_cols:
        df[col] = df[col] * df['log_pop']
    df.drop(['populairity', 'log_pop'], axis=1, inplace=True)
    df = df.resample(time_scale)[sent_cols].agg(['mean', 'std']).fillna(0)
    df.columns = [f"{col}_{stat}" for col, stat in df.columns]
    df.sort_index(inplace=True)
    return df

def load_subreddits(time_scale='h'):
    subs_df = pd.read_csv(REDDIT_SUBS)
    subs_df.dropna(inplace=True)
    subs_list = []
    for _,row in subs_df.iterrows():
        sub_sent_df = process_sentiment(row['Subreddit'], time_scale=time_scale)
        sub_sent_df['team'] = row['Team']
        sub_sent_df = sub_sent_df.round(2)
        subs_list.append(sub_sent_df)

    sub_sent_df = pd.concat(subs_list)
    return sub_sent_df

def get_kalshi_data(time_scale='h'):
    # market_df = pd.read_csv(KALSHI_DATA_HOUR_CSV)
    market_df = pd.read_pickle('data/nfl_historic_candlestick_minute.pkl')
    market_df['team'] = market_df['market'].apply(lambda x: x.split('-')[2])
    market_df['event'] = market_df['market'].apply(lambda x: x.split('-')[1])
    market_df = market_df[['end_period_ts','price.close_dollars','price.high_dollars','price.low_dollars','team','event','volume']].ffill()
    market_df['end_period_ts'] = pd.to_datetime(market_df['end_period_ts'], unit='s')
    market_df = market_df.set_index('end_period_ts')
    market_df = market_df[market_df.index >=pd.to_datetime('2025-08-28')]
    market_df['opp'] = [event[7:].replace(team,'') for event,team in zip(market_df['event'], market_df['team'])]
    market_df.rename(columns={'price.close_dollars':'price_close','price.high_dollars':'price_high','price.low_dollars':'price_low'}, inplace=True)

    # resample by minute, keeping data from each team/event combo
    # also add in back shifted price data
    df_list = []
    for event in market_df['event'].unique():
        event_df = market_df[market_df['event']==event]
        for team in event_df['team'].unique():
            event_df_team = event_df[event_df['team']==team]
            event_df_team = event_df_team.drop_duplicates()
            event_df_team = event_df_team.resample(time_scale).ffill().fillna(0)
            # event_df_team['price_close_prev'] = event_df_team['price_close'].shift(1)
            # event_df_team['price_high_prev'] = event_df_team['price_high'].shift(1)
            # event_df_team['price_low_prev'] = event_df_team['price_low'].shift(1)
            # event_df_team = event_df_team.fillna(0)
            df_list.append(event_df_team)
    market_df = pd.concat(df_list)
    market_df = market_df.reset_index()
    return market_df

def main(time_scale='h'):
    sub_sent_df = load_subreddits(time_scale=time_scale)
    market_df = get_kalshi_data(time_scale=time_scale)

    df = market_df.merge(sub_sent_df, left_on=['end_period_ts','team'], right_on=['time','team'], how='left')
    df = df.merge(sub_sent_df, left_on=['end_period_ts','opp'], right_on=['time','team'], how='left', suffixes=('', '_opp'))
    df['price_close_next'] = df['price_close'].shift(-1)
    df['price_high_next'] = df['price_high'].shift(-1)
    df['price_low_next'] = df['price_low'].shift(-1)
    df.dropna(inplace=True)
    df.set_index('end_period_ts', inplace=True)
    df.sort_index(inplace=True)
    # df.to_csv(f'data/kalshi_reddit_sentiment_combined_{time_scale}.csv', index=True)
    df.to_pickle(f'data/kalshi_reddit_sentiment_combined_{time_scale}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create time series data by combining Kalshi market data with Reddit sentiment data.")
    parser.add_argument('--time_scale', '-ts', type=str, default='h', help="Time scale for resampling data (e.g., 'h' for hourly, 'd' for daily).")
    args = parser.parse_args()
    main(time_scale=args.time_scale)