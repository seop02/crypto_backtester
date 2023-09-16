import ccxt
import numpy as np
import datetime
import time
import logging
import matplotlib.pyplot as plt

def current_prices_binance(currency):
    exchange_binance = ccxt.binance()
    orderbook_binance = exchange_binance.fetch_order_book(currency)
    return orderbook_binance['asks'], orderbook_binance['bids']

def current_prices_upbit(currency):
    exchange_upbit = ccxt.upbit()
    orderbook_upbit = exchange_upbit.fetch_order_book(currency)
    return orderbook_upbit['asks'], orderbook_upbit['bids']

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    orderbook_ask_upbit = []
    orderbook_bid_upbit = []
    orderbook_ask_binance = []
    orderbook_bid_binance = []
    i_values = []
    price_upbit_values = []
    price_binance_values = []
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    i = 0
    currency = 'BTC/USDT'
    currency_upbit = 'BTC/KRW'
    duration = 10
    while i < duration:
        ask_binance, bid_binance = current_prices_binance(currency)
        now_ask_binance = np.array(ask_binance)
        now_bid_binance = np.array(bid_binance)

        ask_upbit, bid_upbit = current_prices_upbit(currency_upbit)
        now_ask_upbit = np.array(ask_upbit)
        now_bid_upbit = np.array(bid_upbit)

        orderbook_ask_upbit.append(now_ask_upbit)
        orderbook_bid_upbit.append(now_bid_upbit)
        orderbook_ask_binance.append(now_ask_binance)
        orderbook_bid_binance.append(now_bid_binance)

        time.sleep(1)
        LOG.info(f'{i}')

        asks_upbit = np.array(orderbook_ask_upbit)
        bids_upbit = np.array(orderbook_bid_upbit)
        asks_binance = np.array(orderbook_ask_binance)
        bids_binance = np.array(orderbook_bid_binance)

        price_binance = ((asks_binance[i][0][0]+bids_binance[i][0][0])/2)
        price_upbit = ((asks_upbit[i][0][0]+bids_upbit[i][0][0])/2)/1320

        LOG.info(price_upbit)

        i_values.append(i)
        price_upbit_values.append(price_upbit)
        price_binance_values.append(price_binance)

        final_upbit = np.concatenate((asks_upbit, bids_upbit), axis=0)
        final_binance = np.concatenate((asks_binance, bids_binance), axis=0)

        i += 1

        #LOG.info(final_upbit.shape)
        #LOG.info(final_binance.shape)
        #np.savetxt(f"{date}_orderbook1.csv", final_upbit.reshape(-1, 200*i))
        #np.save(f'{date}_orderbook1.npy', final_upbit)

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(i_values, price_upbit_values, label='Price Upbit', marker='o')
plt.plot(i_values, price_binance_values, label='Price Binance', marker='o')

plt.xlabel('i')
plt.ylabel('Price')
plt.title('Price Comparison: Upbit vs. Binance')
plt.legend()
plt.grid(True)

plt.show()
