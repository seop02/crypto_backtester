import pandas as pd
from pandas import DataFrame
import numpy as np
from backtest import data_path
import matplotlib.pyplot as plt

class analyzer():
    def import_data(self, date:str, coin:str) -> DataFrame:
        path  = f'{data_path}/ticker/{date}/upbit_volume.parquet'
        df = pd.read_parquet(path)
        coin_array = df['coin'].values
        mask = np.equal(coin_array, coin)
        filtered_data = df[mask]
        return filtered_data
    
    def find_dev(self, date:str, coin:str, threshold:float):
        df = self.import_data(date, coin)
        times = df['time'].values
        devs = df['dev'].values
        prices = df['trade_price'].values
        window_size = 1000
        
        window = np.ones(window_size) / window_size
        moving_avg = np.convolve(np.abs(devs), window, mode='same')
        
        max_length = max(len(devs), len(moving_avg))
        moving_avg = np.pad(moving_avg, (0, max_length - len(moving_avg)), mode='constant', constant_values=0)
        
        signal = np.abs(devs) > threshold
        indices = np.where(signal)[0]
        n = len(indices)
        if n != 0:
        #     for i in range(n):
        #         figure = plt.figure(figsize=(12,6))
        #         time_diff = times[indices[i]:indices[i]+5000]-times[indices[i]]
        #         bought_price = prices[indices[i]]
        #         decaying_price = (1.001/0.9995+0.01*np.exp(-time_diff/100))*bought_price
        #         plt.plot(
        #             times[indices[i]-5000:indices[i]+5000], 
        #             prices[indices[i]-5000:indices[i]+5000], color='black')
        #         plt.plot(times[indices[i]:indices[i]+5000], 
        #                  decaying_price, color='red', linestyle='dashed')
        #         plt.scatter(times[indices[i]], prices[indices[i]], color='blue', s=40)
        #         plt.ylabel('Price')
        #         plt.xlabel('time')
        #         plt.show()
            figure = plt.figure(figsize=(12,6))
            plt.plot(times, prices, color='black')
            plt.scatter(times[signal], prices[signal], color='blue', s=50)
            plt.ylabel('Price')
            plt.xlabel('time')
            plt.show()
            
        else:
            print('NO TRADE!')
            
    def onclick(self, event, labels, data):
        # Left click for class 0
        if event.button == 1:
            labels.append(0)
            print(f'Point {len(labels)} labeled as class 0')
        # Right click for class 1
        elif event.button == 3:
            labels.append(1)
            print(f'Point {len(labels)} labeled as class 1')
        
        # Move to the next point
        if len(labels) < len(data):
            self.plot_point(len(labels))
        else:
            print("Labeling completed!")
            plt.close()

    def plot_point(self, index, data, raw_index, prices, times, decaying_price):
        plt.clf()  # Clear the current figure
        plt.scatter(data[index][0], data[index][1], c='red')  # Highlight the current point in red
        plt.title(f'Label the point {data[index]}')
        plt.draw()  # Draw the plot
            
    def abs_dev(self, df:DataFrame, threshold:float):
        times = df['time'].values
        devs = df['dev'].values
        prices = df['trade_price'].values
        ma = df['trade_price'].rolling(100).mean().values
        window_size = 1000
        
        window = np.ones(window_size) / window_size
        moving_avg = np.convolve(np.abs(devs), window, mode='same')
        
        max_length = max(len(devs), len(moving_avg))
        moving_avg = np.pad(moving_avg, (0, max_length - len(moving_avg)), mode='constant', constant_values=0)
        
        signal = np.abs(devs) > threshold
        indices = np.where(signal)[0]
        n = len(indices)
        if n != 0:
            for i in range(n):
                figure = plt.figure(figsize=(12,6))
                time_diff = times[indices[i]:indices[i]+5000]-times[indices[i]]
                bought_price = prices[indices[i]]
                decaying_price = (1.001/0.9995+0.1*np.exp(-time_diff/1000))*bought_price
                plt.plot(
                    times[indices[i]-5000:indices[i]+5000], 
                    prices[indices[i]-5000:indices[i]+5000], color='black')
                plt.plot(
                    times[indices[i]-5000:indices[i]+5000], 
                    ma[indices[i]-5000:indices[i]+5000], color='orange')
                plt.plot(times[indices[i]:indices[i]+5000], 
                         decaying_price, color='red', linestyle='dashed')
                plt.scatter(times[indices[i]], prices[indices[i]], color='blue', s=40)
                plt.ylabel('Price')
                plt.xlabel('time')
                plt.show()
            figure = plt.figure(figsize=(12,6))
            plt.plot(
                times, prices, color='black')
            plt.scatter(times[signal], prices[signal], color='blue', s=50)
            plt.ylabel('Price')
            plt.xlabel('time')
            plt.show()
            print(len(times))
        else:
            print('NO TRADE!')
            
    def generate_trading_data(self, date:str, coin:str, threshold:float):
        df = self.import_data(date, coin)
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