import pandas as pd
from pandas import DataFrame
import numpy as np
from backtest import data_path
import matplotlib.pyplot as plt

class analyzer():
    def import_data(self, date:str, coin:str, trial:int) -> DataFrame:
        path  = f'{data_path}/ticker/{date}/upbit_volume_{trial}.csv'
        df = pd.read_csv(path, index_col=0)
        coin_array = df['coin'].values
        mask = np.equal(coin_array, coin)
        filtered_data = df[mask]
        return filtered_data
    
    def find_dev(self, date:str, coin:str, threshold:float, trial:int):
        df = self.import_data(date, coin, trial)
        times = df['time'].values
        devs = df['dev'].values
        prices = df['trade_price'].values
        window_size = 1000
        
        window = np.ones(window_size) / window_size
        moving_avg = np.convolve(np.abs(devs), window, mode='same')
        
        max_length = max(len(devs), len(moving_avg))
        moving_avg = np.pad(moving_avg, (0, max_length - len(moving_avg)), mode='constant', constant_values=0)
        
        signal = devs > threshold*moving_avg
        indices = np.where(signal)[0]
        n = len(indices)
        if n != 0:
            for i in range(n):
                figure = plt.figure(figsize=(12,6))
                time_diff = times[indices[i]:indices[i]+5000]-times[indices[i]]
                bought_price = prices[indices[i]]
                decaying_price = (1.001/0.9995+0.01*np.exp(-time_diff/100))*bought_price
                plt.plot(
                    times[indices[i]-5000:indices[i]+5000], 
                    prices[indices[i]-5000:indices[i]+5000], color='black')
                plt.plot(times[indices[i]:indices[i]+5000], 
                         decaying_price, color='red', linestyle='dashed')
                plt.scatter(times[indices[i]], prices[indices[i]], color='blue', s=40)
                plt.ylabel('Price')
                plt.xlabel('time')
                plt.show()
        else:
            print('NO TRADE!')