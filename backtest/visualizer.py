import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas import DataFrame
import pyupbit
from dev_trader import data_path

class generate_plots():
    def plot_transactions(self, data:DataFrame, transaction_idx, date, coin):
        bought_idx = transaction_idx['bought_idx']
        sold_idx = transaction_idx['sold_idx']
        
        bought_price = [data['trade_price'].values[i] for i in bought_idx]
        bought_time = [data['time'].values[i] for i in bought_idx]
        
        sold_price = [data['trade_price'].values[i] for i in sold_idx]
        sold_time = [data['time'].values[i] for i in sold_idx]
        
        figure = plt.figure(figsize=(10,6))
        ax1, ax2 = figure.subplots(2)
        ax1.plot(data['time'], data['trade_price'], color='black')
        ax1.scatter(bought_time, bought_price, color='blue')
        ax1.scatter(sold_time, sold_price, color='red')
        ax2.plot(data['time'], data['dev'], color='black')
        plt.title(f'{date} {coin}')
        plt.show()
    
    def just_plot(self, data:DataFrame):
        figure = plt.figure(figsize=(10,6))
        ax1, ax2 = figure.subplots(2)
        ax1.plot(data['time'], data['trade_price'], color='black')
        # ax1.scatter(bought_time, bought_price, color='blue')
        # ax1.scatter(sold_time, sold_price, color='red')
        ax2.plot(data['time'], data['dev'], color='black')
        plt.show()
        
    def plot_profits(self, coin, daily_profit:dict):
        dates = list(daily_profit.keys())
        dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
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
        ax1.set_title(f'profit for {coin}')
        plt.show()
        
    def plot_ticker(self, coins:list, date):
        file_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        df = pd.read_csv(file_path, index_col=0)
        num = len(coins)

        figure = plt.figure(figsize=(10,6))
        for i, coin in enumerate(coins):
            filtered_data = df[df['coin'] == coin]
            plt.subplot(num, 2, i+1)
            filtered_data['trade_price'].plot()
            plt.xlabel('time')
            plt.ylabel('Price')
            plt.title(f'{coin} price')
            
            plt.subplot(num, 2, i+3)
            filtered_data['dev'].plot()
            plt.xlabel('time')
            plt.ylabel('Price')
            plt.title(f'{coin} dev')
            
        plt.tight_layout()
        plt.show()
        
    

        
        