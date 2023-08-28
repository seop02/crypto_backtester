
import pandas as pd
from binance.client import Client


def current_price(symbol):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
        
    
    client = Client(api_key=api_key, api_secret=secret)
    tickers = client.get_symbol_ticker(symbol=symbol)
    return tickers

def past_daily_price():
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
    
    client = Client(api_key=api_key, api_secret=secret)
    day = client.get_historical_klines(
            symbol="BTCUSDT",
            interval='1s',
            limit=8000)
    return day
        
day = past_daily_price()
print(day[0][8])
print(day[1][8])
print(day[2][8])
print(day[3][8])