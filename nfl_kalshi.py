from datetime import datetime
import pandas as pd
import requests
from utils.auth_utils import sign_pss_text, load_private_key
from utils.constants import KALSHI_ACCESS_KEY


def get_orderbook_with_auth(private_key, path=None):
    current_time = datetime.now()
    timestamp = current_time.timestamp()
    current_time_milliseconds = int(timestamp * 1000)
    timestampt_str = str(current_time_milliseconds)

    method = "GET"
    base_url = 'https://api.elections.kalshi.com'
    # path='/trade-api/v2/markets/KXNFLGAME-25OCT26BUFCAR-BUF/orderbook'
    if path is None:
        path = '/trade-api/v2/markets/KXNFLGAME-25OCT26BUFCAR-BUF/orderbook'

    # Strip query parameters from path before signing
    path_without_query = path.split('?')[0]
    msg_string = timestampt_str + method + path_without_query
    sig = sign_pss_text(private_key, msg_string)

    headers = {
        'KALSHI-ACCESS-KEY': KALSHI_ACCESS_KEY,
        'KALSHI-ACCESS-SIGNATURE': sig,
        'KALSHI-ACCESS-TIMESTAMP': timestampt_str
    }

    response = requests.get(base_url + path, headers=headers, timeout=15)
    return response.json()


def get_historic_nfl_data(time_res='day'):
    private_key = load_private_key()
    path = '/trade-api/v2/markets?limit=1000&series_ticker=KXNFLGAME&status=settled'
    markets = pd.DataFrame(get_orderbook_with_auth(private_key, path)['markets'])

    markets['close_time'] = pd.to_datetime(markets['close_time'], utc=True, format='mixed')
    markets['close_time_ts'] = markets['close_time'].apply(lambda dt: int(dt.timestamp()) if pd.notnull(dt) else None)

    interval_map = {'day': 1440, 'hour': 60, 'minute': 1}
    interval = interval_map[time_res]
    candlestick_list = []
    for _,row in markets.iterrows():
        market = row['ticker']
        end_time = row['close_time_ts']
        start_time = end_time - 604800
        series = market.split('-')[0]
        volume = row['volume']
        path = f'/trade-api/v2/series/{series}/markets/{market}/candlesticks?start_ts={start_time}&end_ts={end_time}&period_interval={interval}'
        orderbook = get_orderbook_with_auth(private_key, path)
        candlesticks = pd.json_normalize(orderbook['candlesticks'])
        candlesticks['market'] = market
        candlesticks['volume'] = volume
        candlestick_list.append(candlesticks)
    print(f"Retrieved {len(candlestick_list)} markets' candlestick data.")
    return pd.concat(candlestick_list)


if __name__ == "__main__":
    df_day = get_historic_nfl_data()
    df_hour = get_historic_nfl_data(time_res='hour')

    with open('data/nfl_historic_candlestick_day.pkl', 'wb') as f:
        pd.to_pickle(df_day, f)
        print("Historic NFL data by day saved as pkl")
    
    with open('data/nfl_historic_candlestick_day.csv', 'w', encoding='utf-8') as f:
        df_day.to_csv(f, index=False)
        print("Historic NFL data by day saved as csv")

    with open('data/nfl_historic_candlestick_hour.pkl', 'wb') as f:
        pd.to_pickle(df_hour, f)
        print("Historic NFL data by hour saved as pkl")
    
    with open('data/nfl_historic_candlestick_hour.csv', 'w', encoding='utf-8') as f:
        df_hour.to_csv(f, index=False)
        print("Historic NFL data by hour saved as csv")