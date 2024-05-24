import pandas as pd
import numpy as np
from backtest import data_path, months
from backtest.generate_training import label_data

if __name__=='__main__':
    coins = ['KRW-BTC', 'KRW-ETC', 'KRW-ETH', 'KRW-GLM', 'KRW-STX', 'KRW-HIVE', 'KRW-XRP']
    dates = ['2024-03', '2024-04']
    X_tot = []
    y_tot = []
    for date in dates:
        for coin in coins:
            print(f'evaluating...{date} {coin}')
            X_sample, y_sample = label_data(coin, date)
            for X, y in zip(X_sample, y_sample):
                X_tot.append(X)
                y_tot.append(y)
    print(X_tot[0:2])
    print(len(y_tot))
    print(y_tot)
    np.save(f'{data_path}/binary/x_more.npy', X_tot)
    np.save(f'{data_path}/binary/y_more.npy', y_tot)