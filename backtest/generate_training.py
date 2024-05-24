import pandas as pd
import numpy as np
from backtest import data_path, months

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = np.array((arr - arr_min) / (arr_max - arr_min))
    return normalized_arr

def label_data(coin, date, threshold=2e2):
    path = f'{data_path}/ticker/{date}/upbit_volume.csv'
    df = pd.read_csv(path, index_col=0)
    df = df[df['coin']==coin]
    times = df['time'].values
    devs = df['dev'].values
    prices = df['trade_price'].values
    ma = df['trade_price'].rolling(1000).mean().values
    
    window_size = 1000
        
    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(np.abs(devs), window, mode='same')
    
    max_length = max(len(devs), len(moving_avg))
    moving_avg = np.pad(moving_avg, (0, max_length - len(moving_avg)), mode='constant', constant_values=0)
    
    signal = np.abs(devs) > threshold*moving_avg
    indices = np.where(signal)[0]
    n = len(indices)
    X = []
    y = []
    if n != 0:
        for i in range(n):
            time_diff = times[indices[i]:]-times[indices[i]]
            bought_price = prices[indices[i]]
            stop_price = 0.99*bought_price
            decaying_price = (1.001/0.9995+0.1*np.exp(-time_diff/500))*bought_price
            good_sell = decaying_price < prices[indices[i]:]
            bad_sell = prices[indices[i]:] < stop_price
            
            good = np.where(good_sell)[0]
            if len(good) == 0:
                good_idx = 0
            else:
                good_idx = good[0]
            bad = np.where(bad_sell)[0] 
            if len(bad) == 0:
                bad_idx = 0
            else:
                bad_idx = bad[0]
                
            if indices[i]>1000:
            
                if good_idx > bad_idx:
                    y.append(0)
                else:
                    y.append(1)
                    
                sub_price = normalize(ma[indices[i]-1000:indices[i]:50])
                X.append(sub_price)
                
    return X, y


    
    
                
    
    
    