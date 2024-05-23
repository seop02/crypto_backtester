import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas import DataFrame
import pyupbit
import numpy as np
from dev_trader import data_path
from scipy.stats import linregress
import math
import plotly.express as px

class generate_plots():
    def is_decreasing_exponential(self,data):
        if len(data) < 10:
            return False
        
        # Take the logarithm of the data
        log_data = [math.log(x) for x in data[-10:]]
        
        # Perform linear regression on log_data
        slope, _, _, _, _ = linregress(range(len(log_data)), log_data)
        
        # Check if the slope is negative
        return slope < -0.5
    
    def plot_transactions(self, data:DataFrame, transaction_idx, date, coin):
        bought_idx = transaction_idx['bought_idx']
        sold_idx = transaction_idx['sold_idx']
        
        bought_price = [data['trade_price'].values[i] for i in bought_idx]
        bought_time = [data['time'].values[i] for i in bought_idx]
        
        sold_price = [data['trade_price'].values[i] for i in sold_idx]
        sold_time = [data['time'].values[i] for i in sold_idx]
        
        ma_list = [1000, 2000, 5000]
        for ma in ma_list:
            data[f'ma{ma}'] = data['trade_price'].rolling(ma).mean()
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        
        axes[0].plot(data['time'], data['trade_price'], color='black')
        axes[0].scatter(bought_time, bought_price, color='blue')
        axes[0].scatter(sold_time, sold_price, color='red')
        axes[1].plot(data['time'], data['dev'], color='black')
        plt.title(f'{date} {coin}')
        plt.show()
    
    def just_plot(self, data:DataFrame):
        ma_list = [1000, 2000, 5000]
        # for ma in ma_list:
        #     name = f'ma{ma}'
        #     data[name] = data['trade_price'].rolling(ma).mean()
            
        figure = plt.figure(figsize=(10,6))
        ax1, ax2 = figure.subplots(2)
        data[['trade_price']].plot(ax=ax1)
        data['dev'].plot(ax=ax2)
        plt.tight_layout()
        plt.show()
        
    def plot_profits(self, daily_profit:dict):
        dates = list(daily_profit.keys())
        #dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
        profits = list(daily_profit.values())
        
        cumulative_profit = []
        init = 1
        for profit in profits:
            init *= profit
            cumulative_profit.append(init)
            
        figure = plt.figure(figsize=(10,6))
        ax1, ax2 = figure.subplots(2)
        ax1.plot(dates, profits, color='black')
        ax2.plot(dates, cumulative_profit, color='black')
        ax1.set_title(f'profit')
        plt.show()
        
    def plot_ticker(self, coins:list, date, trial=None):
        if trial == None:
            file_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        else:
            file_path = f'{data_path}/ticker/{date}/upbit_volume_{trial}.csv'
        df = pd.read_csv(file_path, index_col=0)
        n_rows = len(df)
        segment_size = n_rows // 3
        

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        for i, coin in enumerate(coins):
            coin_array = df['coin'].values
            mask = np.equal(coin_array, 'KRW-BTC')
            filtered_data = df[mask]
            ax1.plot(filtered_data['time'].values, filtered_data['trade_price'].values)
            ax1.set_xlabel('time')
            ax1.set_ylabel('Price')
            ax1.set_title(f'{coin} price')
            
            ax2.plot(filtered_data['time'].values, filtered_data['dev'].values)
            ax2.set_xlabel('time')
            ax2.set_ylabel('dev')
            ax2.set_title(f'{coin} dev')
            
            plt.tight_layout()
            plt.show()
        
    def plot_acc_vol(self, coin:str, date:str, mode:str, duration:float, dev_cut:dict):
        if mode == 'acc':
            path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
            df = pd.read_csv(path, index_col=0)
        elif mode == 'ticker':
            file_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
            df = pd.read_csv(file_path, index_col=0)
            df = df[df['coin'] == coin]
        else:
            raise ValueError('invalid value for mode!')
        
        times = df['time'].values
        start_time = times[0]
        start_vol = df['acc_trade_volume'].values[0]
        sub_acc = []
        tot_acc = []
        tot_idx = []
        
        buy_time = []
        buy_price = []
        
        bought_time = 0
        
        for idx, time in enumerate(times):
            if time-start_time > duration:
                start_time = time
                start_vol = df['acc_trade_volume'].values[idx]      
            acc_vol = df['acc_trade_volume'].values[idx] - start_vol
            if acc_vol < 0:
                start_vol = df['acc_trade_volume'].values[idx]
                acc_vol = df['acc_trade_volume'].values[idx] - start_vol
                start_time = time
                
            sub_acc.append(acc_vol)
            if idx >= 1 and acc_vol-sub_acc[-2] < 0:
                tot_acc.append(sub_acc[-2])
                tot_idx.append(idx-1)
                
            dev = df['dev'].values[idx]
            if dev > dev_cut[coin]:
                print('BUYING!!!')
                bought_time = time
                buy_time.append(time)
                buy_price.append(df['trade_price'].values[idx])
            
        signal_idx = []
        
        for idx, tot_vol in enumerate(tot_acc):
            if idx>=1 and tot_vol/tot_acc[idx-1] < 0.1:
                signal_idx.append(tot_idx[idx])
        
        signal_time = [df['time'].values[i] for i in signal_idx]
        signal_price = [df['trade_price'].values[i] for i in signal_idx]
                
        df['5min_acc'] = np.array(sub_acc)
        figure = plt.figure(figsize=(10,7))
        ax1, ax2, ax3 = figure.subplots(3)
        ax1.plot(df['time'].values, df['trade_price'].values, color='black')
        #ax1.scatter(signal_time, signal_price, color='red')
        ax1.scatter(buy_time, buy_price, color='blue')
        ax2.plot(df['time'].values, df['5min_acc'].values, color='black')
        ax3.plot(df['time'].values, df['dev'].values, color='black')
        ax1.set_title(f'Plot for {coin} {date}')
        plt.show()
        print(tot_acc)
        
                

        
        