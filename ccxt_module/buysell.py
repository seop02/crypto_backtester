import ccxt
import logging
import numpy as np
import orders
import price_data
import pprint

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

with open("api.txt") as f:
    lines = f.readlines()
    key_b = lines[0].strip() 
    secret_b = lines[1].strip()

with open("api_upbit.txt") as f:
    lines = f.readlines()
    key_u = lines[0].strip() 
    secret_u = lines[1].strip()

def last_prices(platform, currency): 
    if platform == 'binance':
        binance = ccxt.binance(config={
        'apiKey': key_b, 
        'secret': secret_b,
        'enableRateLimit': True
        })
        binance_market = binance.fetch_ticker(currency)
        price = binance_market['last']
    
    elif platform == 'binance_future':
        binance = ccxt.binance(config={
        'apiKey': key_b, 
        'secret': secret_b,
        'enableRateLimit': True,
         'options': {
            'defaultType': 'future'
        }
        })
        binance_market = binance.fetch_ticker(currency)
        price = binance_market['last']

    elif platform == 'upbit':
        upbit = ccxt.upbit(config={
        'apiKey': key_u, 
        'secret': secret_u,
        'enableRateLimit': True
        })
        upbit_market = upbit.fetch_ticker(currency)
        price = upbit_market['last']
    
    return price

def current_amount(currency, platform):
    if platform == 'upbit':
        exchange = ccxt.upbit(config={
        'apiKey': key_u,
        'secret': secret_u
        }
        )

        # balance
        balance = exchange.fetch_balance()
    elif platform == 'binance':
        exchange = ccxt.binance(config={
        'apiKey': key_b,
        'secret': secret_b
        }
        )

        # balance
        balance = exchange.fetch_balance()
        
    if len(balance['info']) == 1:
        usd_balance = balance['KRW']
        xrp_balance = {'free': 0.0, 'used': 0.0, 'total': 0.0}
    else:
        usd_balance = balance['KRW']
        xrp_balance = balance[currency]

    LOG.info(f"usd_balance: {usd_balance}")
    LOG.info(f"ripple_balance: {xrp_balance}")
    return xrp_balance, usd_balance

def buy_order(currency, amount, price):
    binance = ccxt.upbit(config={
        'apiKey': key_u,
        'secret': secret_u,
        'enableRateLimit': True
    })
    price = round(price, 1)
    order = binance.create_limit_buy_order(
        symbol=currency, 
        amount=amount,
        price=price
    )
    pprint.pprint(order)
    return order

def sell_order(currency, amount, price):

    binance = ccxt.upbit(config={
        'apiKey': key_u,
        'secret': secret_u,
        'enableRateLimit': True
    })
    
    price = round(price, 1)
    order = binance.create_limit_sell_order(
        symbol=currency, 
        amount=amount,
        price = price
    )
    pprint.pprint(order)
    return order

def cancel_order(currency, order_id):
    #cancel order 

    # binance 객체 생성
    binance = ccxt.binance(config={
        'apiKey': key_u,
        'secret': secret_u
    })

    resp = binance.cancel_order(
        id=order_id,
        symbol=currency
    )
    return resp


def current_prices(currency, platform):
    if platform == 'binance':
        exchange = ccxt.binance()
        orderbook = exchange.fetch_order_book(currency)
        return orderbook['asks'], orderbook['bids']
    elif platform == 'upbit':
        exchange = ccxt.upbit()
        orderbook = exchange.fetch_order_book(currency)
        return orderbook['asks'], orderbook['bids']

try:
    binance = 'binance'
    binancef = 'binance_future'
    upbit = 'upbit'

    currency = 'DOGE'
    currency_us = 'DOGE/USDT'
    currency_kr = 'DOGE/KRW'

    binance_price = []
    upbit_price = []
    LOG.info('start')
    cry_balance, usd_balance = current_amount(currency, upbit)
    current_ask, current_bid = current_prices(currency_kr, upbit)
    current_price = (current_ask[0][0] + current_bid[0][0])/2
    asset = usd_balance['total'] + current_price*cry_balance['total']
    LOG.info(asset)
    duration = 0
    orderID = 0
    LOG.info(asset)
    while asset > 100000:
        LOG.info('-----------------')
        LOG.info(f'current asset: {asset}')

        if cry_balance['used'] != 0 or usd_balance['used'] != 0:
            LOG.info('cancelling the existing orders')
            cancel_order(currency_kr, orderID)


        usd = last_prices(binancef, currency_us)
        krw = last_prices(upbit, currency_kr)
        LOG.info(f'current binance price: {usd}')
        binance_price.append(usd)
        
        if len(binance_price)>=2:
            diff = (binance_price[duration]-binance_price[duration-1])/binance_price[duration-1]
            cry_balance, usd_balance = current_amount(currency, upbit)
            current_ask, current_bid = current_prices(currency_kr, upbit)
            current_price = (current_ask[0][0] + current_bid[0][0])/2
            LOG.info(diff)
            diff = 2
            if diff > 0.0006:
                if usd_balance['free'] <= 5000:
                    pass
                else:
                    LOG.info('buying')
                    current_price = round(current_price,1)
                    balance = round(usd_balance['free'], 1)
                    amount = usd_balance['free']/current_price
                    buy = buy_order(currency_kr, amount, current_price)
                    orderID = buy['id']
                    cry_balance, usd_balance = current_amount(currency, upbit)
            elif diff < -0.0004:
                if cry_balance['free'] <= 60:
                    pass
                else:
                    LOG.info('selling')
                    amount = cry_balance['free']
                    sell = sell_order(currency_kr, amount, current_price)
                    orderID = sell['id']
                    cry_balance, usd_balance = current_amount(currency, upbit)
            
        duration += 1
except:
    LOG.info("*********************error*********************")
    cry_balance, usd_balance = current_amount(currency, upbit)
else:
    LOG.info("=====================done=====================")
    cry_balance, usd_balance = current_amount(currency, upbit)
    
 