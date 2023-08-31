import ccxt
import numpy as np
import datetime
import time
import logging

def current_prices(currency):
    exchange = ccxt.binance()
    orderbook = exchange.fetch_order_book(currency)
    return orderbook['asks'], orderbook['bids']

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    orderbook_ask = []
    orderbook_bid = []
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    i = 0
    currency = 'XRP/USDT'
    duration = 86400
    while i < duration:
        ask, bid = current_prices(currency)
        now_ask = np.array(np.transpose(ask))
        now_bid = np.array(np.transpose(bid))

        orderbook_ask.append(now_ask)
        orderbook_bid.append(now_bid)
        i += 1
        LOG.info(f'{i}')

        asks = np.array(orderbook_ask)
        bids = np.array(orderbook_bid)

        final = np.concatenate((asks, bids), axis=1)
        LOG.info(final.shape)
        np.savetxt(f"{date}_orderbook1.csv", final.reshape(-1, 200*i))
        np.save(f'{date}_orderbook.npy', final)


