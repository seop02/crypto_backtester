import pprint
from binance.client import Client



def get_orderbook(symbol):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()

    client = Client(api_key=api_key, api_secret=secret)
    orderbook = client.get_order_book(symbol='BTCUSDT')
    
    return orderbook

def get_balance(asset):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
    
    client = Client(api_key=api_key, api_secret=secret)
    balance = client.get_asset_balance(asset=asset)
    return balance

