import os
from dotenv import load_dotenv

load_dotenv()

KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USER_AGENT = "data-collection for sentiment analysis:1.0 (by u/Mysterious-Mobile-69)"

KALSHI_DATA_DAY = "data/nfl_hisoric_candlestick_day.pkl"
KALSHI_DATA_HOUR = "data/nfl_hisoric_candlestick_hour.pkl"
KALSHI_DATA_MINUTE = "data/nfl_hisoric_candlestick_minute.pkl"
KALSHI_DATETIME_COLS = ['end_period_ts']
KALSHI_FEATURE_COLS = [ 
                        'price_close', 'price_high', 'price_low', 'volume',
                        'oai_negative_mean', 'oai_negative_std', 
                        'oai_neutral_mean','oai_neutral_std',
                        'oai_positive_mean', 'oai_positive_std',
                        'vader_neg_mean', 'vader_neg_std', 
                        'vader_neu_mean', 'vader_neu_std',
                        'vader_pos_mean', 'vader_pos_std', 
                        'oai_negative_mean_opp','oai_negative_std_opp', 
                        'oai_neutral_mean_opp', 'oai_neutral_std_opp',
                        'oai_positive_mean_opp', 'oai_positive_std_opp', 
                        'vader_neg_mean_opp','vader_neg_std_opp', 
                        'vader_neu_mean_opp', 'vader_neu_std_opp',
                        'vader_pos_mean_opp', 'vader_pos_std_opp',
                        ]
KALSHI_FEATURE_COLS_OAI = [ 
                        'price_close', 'price_high', 'price_low', 'volume',
                        'oai_negative_mean', 'oai_negative_std', 
                        'oai_neutral_mean','oai_neutral_std',
                        'oai_positive_mean', 'oai_positive_std',
                        'oai_negative_mean_opp','oai_negative_std_opp', 
                        'oai_neutral_mean_opp', 'oai_neutral_std_opp',
                        'oai_positive_mean_opp', 'oai_positive_std_opp', 
                        ]
KALSHI_FEATURE_COLS_VADER = [ 
                        'price_close', 'price_high', 'price_low', 'volume',
                        'vader_neg_mean', 'vader_neg_std', 
                        'vader_neu_mean', 'vader_neu_std',
                        'vader_pos_mean', 'vader_pos_std', 
                        'vader_neg_mean_opp','vader_neg_std_opp', 
                        'vader_neu_mean_opp', 'vader_neu_std_opp',
                        'vader_pos_mean_opp', 'vader_pos_std_opp',
                        ]
REDDIT_DATA_PATH = "data/subs_sentiment"
KALSHI_DATA_DAY_CSV = "data/nfl_historic_candlestick_day.csv"
KALSHI_DATA_HOUR_CSV = "data/nfl_historic_candlestick_hour.csv"
REDDIT_SUBS = 'data/reddit_subs.csv'
