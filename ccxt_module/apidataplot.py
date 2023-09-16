import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def plotapi(title, upbitcu):


    bin = pd.read_csv(f'apidata_{title}_{upbitcu}_binance.csv')
    upb = pd.read_csv(f'apidata_{title}_{upbitcu}_upbit.csv')

    price_binance = bin['last'].values * upb['last'].values[0]/bin['last'].values[0]
    time_binance = bin['timestamp'].values/1000 - bin['timestamp'].values[0]/1000

    price_upbit = upb['last'].values
    time_upbit = upb['timestamp'].values/1000 -bin['timestamp'].values[0]/1000


    plt.figure(figsize=(10, 6))
    plt.plot(time_binance, price_binance, label='Price Binance', marker='o')
    plt.plot(time_upbit, price_upbit, label='Price Upbit', marker='o')

    plt.xlabel('Time (second)')
    plt.ylabel('Price (USD)')
    plt.title(f'Price Comparison of {title}: Upbit vs. Binance')
    plt.legend()
    plt.grid(True)

    plt.show()


plotapi('storj', 'krw')