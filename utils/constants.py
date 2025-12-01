import os
from dotenv import load_dotenv

load_dotenv()

KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY")
KALSHI_DATA_DAY = "data/nfl_hisoric_candlestick_day.pkl"
KALSHI_DATA_HOUR = "data/nfl_hisoric_candlestick_hour.pkl"
KALSHI_DATETIME_COLS = ['end_period_ts']
REDDIT_DATA_PATH = "data/subs_sentiment"