import sys
sys.path.append("./")


import ccxt
import pprint


def insert_order(currency, amount, price):
    with open("ccxt_module/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip() 

    binance = ccxt.binance(config={
        'apiKey': api_key,
        'secret': secret
    })

    order = binance.create_limit_buy_order(
        symbol=currency, 
        amount=amount, 
        price=price
    )
    pprint.pprint(order)
    return order

def sell_order(currency, amount, price):
    with open("ccxt_module/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip() 

    binance = ccxt.binance(config={
        'apiKey': api_key,
        'secret': secret
    })

    order = binance.create_limit_sell_order(
        symbol=currency, 
        amount=amount, 
        price=price
    )
    pprint.pprint(order)
    return order

def current_balance():
    with open("ccxt_module/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        api_secret = lines[1].strip() 
    exchange = ccxt.binance(config={
    'apiKey': api_key,
    'secret': api_secret
    }
    )

    # balance
    balance = exchange.fetch_balance()
    return balance

def cancel_order(currency, order_id):
    #cancel order
    with open("ccxt_module/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip() 
        secret = lines[1].strip() 

    # binance 객체 생성
    binance = ccxt.binance(config={
        'apiKey': api_key,
        'secret': secret
    })

    resp = binance.cancel_order(
        id=order_id,
        symbol=currency
    )
    return resp