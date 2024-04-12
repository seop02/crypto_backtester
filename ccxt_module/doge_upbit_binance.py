import ccxt
import numpy as np
import datetime
import time
import logging

def current_prices(currency):
    exchange = ccxt.binance()
    orderbook = exchange.fetch_order_book(currency)
    return orderbook['asks'], orderbook['bids']

def current_prices_upbit(currency):
    exchange = ccxt.upbit()
    orderbook = exchange.fetch_order_book(currency)
    return orderbook['asks'], orderbook['bids']

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


def data_collection(currency, upbit_currency):
    orderbook_ask = []
    orderbook_bid = []
    orderbook_ask_upbit = []
    orderbook_bid_upbit = []
    time_binance = []
    time_upbit = []

    now = datetime.datetime.now()
    date = now.strftime("%m/%d/%Y, %H:%M:%S")
    i = 0
    duration = 600

    while i < duration:

        current_time_in_seconds_upbit = time.time()
        current_time_in_milliseconds_upbit = current_time_in_seconds_upbit * 1000
        ask_upbit, bid_upbit = current_prices_upbit(upbit_currency)
        now_ask_upbit = np.array(ask_upbit)
        now_bid_upbit = np.array(bid_upbit)
        time_upbit.append(current_time_in_milliseconds_upbit)

        current_time_in_seconds_binance = time.time()
        current_time_in_milliseconds_binance = current_time_in_seconds_binance * 1000
        ask, bid = current_prices(currency)
        now_ask = np.array(ask)
        now_bid = np.array(bid)
        time_binance.append(current_time_in_milliseconds_binance)


        orderbook_ask.append(now_ask)
        orderbook_bid.append(now_bid)

        orderbook_ask_upbit.append(now_ask_upbit)
        orderbook_bid_upbit.append(now_bid_upbit)

        LOG.info(f'{i}')

        asks = np.array(orderbook_ask)
        bids = np.array(orderbook_bid)
        asks_upbit = np.array(orderbook_ask_upbit)
        bids_upbit = np.array(orderbook_bid_upbit)

        i += 1


        final = np.concatenate((asks, bids), axis=0)
        final_upbit = np.concatenate((asks_upbit, bids_upbit), axis=0)
        LOG.info(final.shape)
        LOG.info(final_upbit.shape)
        np.save(f'{currency}_orderbook1.npy', final)
        np.save(f'doge_upbit_orderbook1.npy', final_upbit)
        np.save(f'doge_time_binance.npy', time_binance)
        np.save(f'doge_time_upbit.npy', time_upbit)