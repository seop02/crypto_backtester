import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def odb_plot(title, upbitcu):
    data = np.load(f'odb_{title}_{upbitcu}_orderbook1.npy')
    data_upbit = np.load(f'odb_{title}_{upbitcu}_upbit_orderbook1.npy')

    time_binance = np.load(f'odb_{title}_{upbitcu}_time_binance.npy')
    time_upbit = np.load(f'odb_{title}_{upbitcu}_time_upbit.npy')

    time_binance = (time_binance - time_upbit[0])/1000
    time_upbit = (time_upbit - time_upbit[0])/1000



    X_raw = data
    X_raw = np.array(X_raw)
    length = int(X_raw.shape[0]/2)

    Y_raw = data_upbit
    Y_raw = np.array(Y_raw)
    length_upbit = int(Y_raw.shape[0]/2)

    LOG.info("Upbit")
    LOG.info(Y_raw[1])
    LOG.info("Binance")
    LOG.info(X_raw[1])
    

    idx_values = []
    idx_values_upbit = []


    price = []
    for idx in range(length):
        mid = (X_raw[idx][0][0] + X_raw[idx+length][2][0])/2
        price.append(mid)
        idx_values.append(idx)

    price = np.array(price)

    price_upbit = []
    for idx_upbit in range(length):
        mid_upbit = (Y_raw[idx_upbit][0][0] + Y_raw[idx_upbit+length_upbit][2][0])/2
        price_upbit.append(mid_upbit)
        idx_values_upbit.append(idx_upbit)

    price_upbit = np.array(price_upbit)
    price = price*price_upbit[0]/price[0]


    plt.figure(figsize=(10, 6))
    plt.plot(time_binance, price, label='Price Binance', marker='o')
    plt.plot(time_upbit, price_upbit, label='Price Upbit', marker='o')

    plt.xlabel('Time (second)')
    plt.ylabel(f'Price ({upbitcu})')
    plt.title(f'Price Comparison of {title}: Upbit vs. Binance')
    plt.legend()
    plt.grid(True)

    plt.show()


odb_plot('storj', 'krw')