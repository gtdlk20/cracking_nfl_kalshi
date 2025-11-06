import pandas as pd
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
import requests
import json
from datetime import datetime
import requests

KALSHI_PRIVATE_KEY_PATH = 'auth.json'
KALSHI_ACCESS_KEY = "163cfa47-6144-4378-8d93-4b683341d760"

def load_private_key_from_file(file_path):
    with open(file_path, "rb") as key_file:
        data = json.load(key_file)
        private_key = serialization.load_pem_private_key(
            data['kalshi_api_key'].encode('utf-8'),
            password=None,  # or provide a password if your key is encrypted
            backend=default_backend()
        )
        print("Private key loaded successfully.")
    return private_key

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        print("Message signed successfully.")
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        raise ValueError("RSA sign PSS failed") from e
        print("Failed to sign message.")
    

def get_orderbook_with_auth(path=None):
    current_time = datetime.now()
    timestamp = current_time.timestamp()
    current_time_milliseconds = int(timestamp * 1000)
    timestampt_str = str(current_time_milliseconds)

    private_key = load_private_key_from_file(KALSHI_PRIVATE_KEY_PATH)

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

    response = requests.get(base_url + path, headers=headers)
    return response.json()


def get_historic_nfl_data(time_res='day'):
    path = '/trade-api/v2/markets?limit=1000&series_ticker=KXNFLGAME&status=settled'
    markets = pd.DataFrame(get_orderbook_with_auth(path)['markets'])
    with open('nfl_historic_markets_data.pkl', 'wb') as f:
        pd.to_pickle(markets, f)
        print("Historic NFL markets data saved as pkl")
    
    with open('nfl_historic_markets_data.csv', 'w') as f:
        markets.to_csv(f, index=False)
        print("Historic NFL markets data saved as csv")

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
        path = f'/trade-api/v2/series/{series}/markets/{market}/candlesticks?start_ts={start_time}&end_ts={end_time}&period_interval={interval}'
        candlestick_list.append(pd.DataFrame(get_orderbook_with_auth(path)['candlesticks']))
    print(f"Retrieved {len(candlestick_list)} markets' candlestick data.")
    return pd.concat(candlestick_list)

if __name__ == "__main__":
    df = get_historic_nfl_data()
    print(df.head())
    with open('nfl_historic_candlestick_data.pkl', 'wb') as f:
        pd.to_pickle(df, f)
        print("Historic NFL data saved as pkl")
    
    with open('nfl_historic_candlestick_data.csv', 'w') as f:
        df.to_csv(f, index=False)
        print("Historic NFL data saved as csv")