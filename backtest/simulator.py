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
        self.daily_profits = {date: 0 for date in dates}
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
        
    def import_individual_data(self, coin, date, data_type) -> DataFrame:
        if data_type == 'acc':   
            path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
            data = pd.read_csv(path, index_col=0)
        elif data_type == 'ticker':
            path = f'{data_path}/ticker/{date}/upbit_volume.csv'
            raw_data = pd.read_csv(path, index_col=0)
            data = raw_data[raw_data['coin'] == coin]
        return data
    
    def import_all(self, date):
        path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        df = pd.read_csv(path, index_col=0)
        return df
    
    def simulate(self, df:DataFrame, dev_cut, profit_cut, date) -> float:      
        times = df['time'].values
        vol = df['acc_trade_volume'].values
        devs = df['dev'].values
        price = df['trade_price'].values
        highs = df['high'].values
        status = 'sold'
        
        
        buying_price = 0
        max_profit = 0
        profit = 1
        
        for idx, dev in enumerate(devs):
            if status != "sold" and buying_price != 0:
                inst_profit = (self.transaction*price[idx]/buying_price)
                max_price= price[idx]
                max_idx = idx
                max_profit = max(inst_profit, max_profit)
            else:
                inst_profit = 1
                
            if idx > 5:
                mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                
            condition_1 = dev>=dev_cut and status == 'sold' and idx>5
            # if dev>dev_cut:
            #     print('HIIIII')
            #     print(self.status)
            #     print(idx)
            if condition_1 == True:
                status = 'buying'
                buying_price = price[idx]
                self.transaction_times[f'{status}_times'].append(times[idx])
                self.transaction_idx[date][f'{status}_idx'].append(idx)
                
            if status == 'buying' and buying_price >= price[idx]:
                status = 'bought'
                self.transaction_times[f'{status}_times'].append(times[idx])
                self.transaction_idx[date][f'{status}_idx'].append(idx)
                
            if status == 'bought' and inst_profit <= -0.05:
                status = 'selling'
                selling_price = price[idx]
                self.transaction_times[f'{status}_times'].append(times[idx])
                self.transaction_idx[date][f'{status}_idx'].append(idx)
                
            if status == 'bought' and inst_profit>=profit_cut:
                status = 'selling'
                selling_price = price[idx]
                self.transaction_times[f'{status}_times'].append(times[idx])
                self.transaction_idx[date][f'{status}_idx'].append(idx)
                
            if status == 'selling' and selling_price <= price[idx]:
                status = 'sold'
                self.transaction_times[f'{status}_times'].append(times[idx])
                self.transaction_idx[date][f'{status}_idx'].append(idx)
                inst_profit = (self.transaction*price[idx]/buying_price)
                profit *= inst_profit
                max_profit = 0
                
            if status == 'buying':
                time_diff = times[idx]-self.transaction_times[f'{status}_times'][-1]
                if time_diff > 30:
                    status = 'sold'
            
            if status == 'selling':
                time_diff = times[idx]-self.transaction_times[f'{status}_times'][-1]
                if time_diff > 30:
                    status = 'bought'
        if status == 'bought':
            status = 'sold'
            self.transaction_times[f'{status}_times'].append(times[idx])
            self.transaction_idx[date][f'{status}_idx'].append(idx)
            inst_profit = (self.transaction*price[idx]/buying_price)
            profit *= inst_profit
            max_profit = 0

        return profit
            
    def update_profit(self, dev_cut, profit_cut, date, coin, data_type):
        profit = self.simulate(dev_cut, profit_cut, date, data_type, coin)
        LOG.info(f'{coin} PROFIT for {date}: {self.daily_profits[date]}')
    
    def individual_simulation(self, coin:str, dev_cut:dict, profit_cut:dict):
        vis = generate_plots()
        for date in self.dates:
            if date == '2024-05-03':
                mode = 'ticker'
            else:
                mode = 'acc'
            df = self.import_individual_data(coin, date, mode)
            profit = self.simulate(df, dev_cut[coin], profit_cut[coin], date)
            vis
            self.daily_profits[date] = profit
            if profit != 1:
                vis.plot_transactions(df, self.transaction_idx[date], date, coin)
            else:
                vis.just_plot(df)
            
            self.transaction_idx = {
            date : {f'{status}_idx': [] for status in self.statuses} for date in self.dates
            }
                
        vis.plot_profits(coin, self.daily_profits)
        
    def simulate_all(self, date:str, dev_cut:dict, profit_cut:dict):
        df = self.import_all(date)
        times = df['time'].values
        vol = df['acc_trade_volume'].values
        devs = df['dev'].values
        price = df['trade_price'].values
        highs = df['high'].values
        coins = df['coin'].values
        trading_coins = set(dev_cut.keys())
        status = 'sold'
        
        buying_price = 0
        max_profit = 0
        profit = 1
        
        for idx, dev in enumerate(devs):
            coin = coins[idx]
            if status != "sold" and buying_price != 0:
                inst_profit = (self.transaction*price[idx]/buying_price)
                max_profit = max(inst_profit, max_profit)
            else:
                inst_profit = 1
                
            if idx > 5:
                mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                
            condition_1 = coin in trading_coins and dev>=dev_cut[coin] and status == 'sold' and idx>5
            # if dev>dev_cut:
            #     print('HIIIII')
            #     print(self.status)
            #     print(idx)
            if condition_1 == True:
                status = 'buying'
                buying_price = price[idx]
                
            if status == 'buying' and buying_price >= price[idx]:
                status = 'bought'
                
            if status == 'bought' and inst_profit <= -0.05:
                status = 'selling'
                selling_price = price[idx]
                
            if status == 'bought' and inst_profit>=profit_cut[coin]:
                status = 'selling'
                selling_price = price[idx]
                
            if status == 'selling' and selling_price <= price[idx]:
                status = 'sold'
                inst_profit = (self.transaction*price[idx]/buying_price)
                profit *= inst_profit
                max_profit = 0
                
            if status == 'buying':
                time_diff = times[idx]-self.transaction_times[f'{status}_times'][-1]
                if time_diff > 30:
                    status = 'sold'
            
            if status == 'selling':
                time_diff = times[idx]-self.transaction_times[f'{status}_times'][-1]
                if time_diff > 30:
                    status = 'bought'
        if status == 'bought':
            status = 'sold'
            inst_profit = (self.transaction*price[idx]/buying_price)
            profit *= inst_profit
            max_profit = 0

        return profit
        
            
    
    
        
            
        
    
    