import ccxt

exchange = ccxt.binance()
orderbook = exchange.fetch_order_book('ETH/USDT')
print(orderbook['asks'])
print(orderbook['bids'])

#returns current [price, volume]

def current_prices(currency):
    exchange = ccxt.binance()
    orderbook = exchange.fetch_order_book(currency)
    return orderbook['asks'], orderbook['bids']

