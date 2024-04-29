import pyupbit
import pandas as pd
import numpy as np
from dev_trader import data_path
import math
import os
import logging
from pandas import DataFrame
from backtest.visualizer import generate_plots

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class backtrader():
    def __init__(self, coins, dates):
        self.coins = coins
        self.dates = dates
    
        self.transaction = 0.9995**2
        self.profit = 1
        self.statuses = ['buying', 'bought', 'selling', 'sold']
        self.daily_profits = {coin: {date: 0 for date in dates} for coin in coins}
        self.transaction_times = {f'{status}_times': [] for status in self.statuses}
        self.transaction_idx = {
            date : {f'{status}_idx': [] for status in self.statuses} for date in dates
        }
        
        
    def return_step(self, coin, price):
        step = {}
        if price<0.01:
            step[coin] = 0.000001
        elif 0.01<=price<0.1:
            step[coin] = 0.00001
        elif 1<price<10:
            step[coin] = 0.001
        elif 10<=price<100:
            step[coin] = 0.01
        elif 100<=price<1000:
            step[coin] = 0.1
        elif 1000<=price<10000:
            step[coin] = 1
        elif 10000<=price<100000:
            step[coin] = 10
        elif 100000<=price<1000000:
            step[coin] = 50
        elif 1000000<=price:
            step[coin] = 1000
        #LOG.info(f'updating step of {coin} to {step[coin]}')
        return step[coin]

    def round_sigfigs(self, num, sig_figs):
        if num != 0:
            return round(num, -int(math.floor(math.log10(abs(num)))) + (sig_figs - 1))
        else:
            return 0.0
        
    def import_data(self, coin, date) -> DataFrame:
        path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
        data = pd.read_csv(path, index_col=0)
        return data
    
    def simulate(self, dev_cut, profit_cut, date, coin):
        file_path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            times = df['time'].values
            vol = df['acc_trade_volume'].values
            devs = df['dev'].values
            price = df['trade_price'].values
            highs = df['high'].values
            ask_bid = df['ask_bid'].values
            
            
            self.status = 'sold'
            
            buying_price = 0
            max_profit = 0
            profit = 1
            
            for idx, dev in enumerate(devs):
                if self.status != "SOLD" and buying_price != 0:
                    inst_profit = (self.transaction*price[idx]/buying_price)-1
                    max_price= price[idx]
                    max_idx = idx
                    max_profit = max(inst_profit, max_profit)
                else:
                    inst_profit = 0
                    
                if idx > 5:
                    mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                    
                condition_1 = dev>dev_cut and self.status == 'sold' and idx>5
                # if dev>dev_cut:
                #     print('HIIIII')
                #     print(self.status)
                #     print(idx)
                if condition_1 == True:
                    self.status = 'buying'
                    buying_price = price[idx]
                    self.transaction_times[f'{self.status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    
                if self.status == 'buying' and buying_price >= price[idx]:
                    self.status = 'bought'
                    self.transaction_times[f'{self.status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    
                if self.status == 'bought' and inst_profit <= -0.05:
                    self.status = 'selling'
                    selling_price = price[idx]
                    self.transaction_times[f'{self.status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    
                # if self.status == 'bought' :
                #     self.status = 'selling'
                #     selling_price = price[idx]
                #     self.transaction_times[f'{self.status}_times'].append(times[idx])
                #     self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    
                if self.status == 'bought' and inst_profit>profit_cut:
                    self.status = 'selling'
                    selling_price = price[idx]
                    self.transaction_times[f'{self.status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    
                if self.status == 'selling' and selling_price <= price[idx]:
                    self.status = 'sold'
                    self.transaction_times[f'{self.status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    profit *= inst_profit
                    max_profit = 0
                    
                if self.status == 'buying':
                    time_diff = times[idx]-self.transaction_times[f'{self.status}_times'][-1]
                    if time_diff > 30:
                        self.status = 'sold'
                
                if self.status == 'selling':
                    time_diff = times[idx]-self.transaction_times[f'{self.status}_times'][-1]
                    if time_diff > 30:
                        self.status = 'bought'
            if self.status == 'bought':
                self.status = 'sold'
                self.transaction_times[f'{self.status}_times'].append(times[idx])
                self.transaction_idx[date][f'{self.status}_idx'].append(idx)
                inst_profit = (self.transaction*price[idx]/buying_price)
                profit *= inst_profit
                max_profit = 0
        
        else:
            print(f'{file_path} does not exists!!!')
        
        self.daily_profits[coin][date] = profit
            
    def update_profit(self, dev_cut, profit_cut, date, coin):
        self.simulate(dev_cut, profit_cut, date, coin)
        LOG.info(f'{coin} PROFIT for {date}: {self.daily_profits[coin][date]}')
    
    def run_simulation(self, coins, dev_cut, profit_cut, mode):
        vis = generate_plots()
        for coin in coins:
            for date in self.dates:
                self.update_profit(dev_cut[coin], profit_cut, date, coin)
                
                if len(self.transaction_idx[date]['bought_idx']) != 0 and mode == True:
                    vis.plot_transactions(date, coin, self.transaction_idx[date])
                elif mode == True:
                    vis.just_plot(date, coin)
            if self.daily_profits[coin][date] != 0:
                vis.plot_profits(coin, self.daily_profits[coin])
            for status in self.statuses:
                self.transaction_idx[date][f'{status}_idx'] = []
    
    
        
            
        
    
    