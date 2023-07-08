from binance.client import Client
import pprint

def buy_order(asset, price, volume):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
    
    client = Client(api_key=api_key, api_secret=secret)
    order = client.order_limit_buy(
            symbol= asset, quantity = volume, price = price
    )
    print(order)

def sell_order(asset, price, volume):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
    
    client = Client(api_key=api_key, api_secret=secret)
    order = client.order_limit_sell(
            symbol= asset, quantity = volume, price = price
    )
    print(order)
    
def cancel_order(asset, orderId):
    with open("pb/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip()
    
    client = Client(api_key=api_key, api_secret=secret)
    order = client.cancel_order(
            symbol= asset, orderId=orderId
    )
    print(order)