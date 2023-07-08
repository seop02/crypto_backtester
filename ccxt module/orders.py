import ccxt
import pprint

#insert order
with open("api.txt") as f:
    lines = f.readlines()
    api_key = lines[0].strip() 
    secret = lines[1].strip() 

binance = ccxt.binance(config={
    'apiKey': api_key,
    'secret': secret
})

order = binance.create_limit_buy_order(
    symbol="BTC/USDT", 
    amount=0.01, 
    price=20000
)

pprint.pprint(order)

#cancel order
with open("api.txt") as f:
    lines = f.readlines()
    api_key = lines[0].strip() 
    secret = lines[1].strip() 

# binance 객체 생성
binance = ccxt.binance(config={
    'apiKey': api_key,
    'secret': secret
})

resp = binance.cancel_order(
    id=5221422745,
    symbol='BTC/USDT'
)

pprint.pprint(resp)

def insert_order(currency, amount, price):
    with open("api.txt") as f:
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

def cancel_order(currency, order_id):
    #cancel order
    with open("api.txt") as f:
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