import pyupbit
import pandas as pd
import numpy as np
from dev_trader import data_path
import math
import os
import logging
from pandas import DataFrame
from backtest.visualizer import generate_plots
from scipy.stats import linregress
import math
from datetime import datetime
import pyupbit

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
        self.step = {coin: 0 for coin in coins}
    
    def compute_sharpe_ratio(self, profits):
        # Calculate the mean profit
        mean_profit = sum(profits) / len(profits)

        # Calculate the standard deviation of profits
        squared_diffs = [(profit - mean_profit) ** 2 for profit in profits]
        variance = sum(squared_diffs) / len(profits)
        std_dev = math.sqrt(variance)

        # Compute the Sharpe ratio
        sharpe_ratio = mean_profit / std_dev if std_dev != 0 else 0

        return sharpe_ratio
        
        
    def return_step(self, price):
        step = 0
        if price<0.01:
            step = 0.000001
        elif 0.01<=price<0.1:
            step = 0.00001
        elif 1<price<10:
            step = 0.001
        elif 10<=price<100:
            step = 0.01
        elif 100<=price<1000:
            step = 0.1
        elif 1000<=price<10000:
            step = 1
        elif 10000<=price<100000:
            step = 10
        elif 100000<=price<1000000:
            step = 50
        elif 1000000<=price:
            step = 1000
        #LOG.info(f'updating step of {coin} to {step[coin]}')
        return step

    def round_sigfigs(self, num, sig_figs):
        if num != 0:
            return round(num, -int(math.floor(math.log10(abs(num)))) + (sig_figs - 1))
        else:
            return 0.0
        
    def import_individual_data(self, coin, date) -> DataFrame:
        default_date = datetime.strptime('2024-04-28', '%Y-%m-%d')
        input_date = datetime.strptime(date, '%Y-%m-%d')
        
        if input_date<default_date:   
            path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
            data = pd.read_csv(path, index_col=0)
        else:
            path = f'{data_path}/ticker/{date}/upbit_volume.csv'
            raw_data = pd.read_csv(path, index_col=0)
            data = raw_data[raw_data['coin'] == coin]
            data = data.reset_index(drop=True, level=None)
        return data
    
    def import_all(self, date):
        path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        df = pd.read_csv(path, index_col=0)
        return df
    
    def is_decreasing_exponential(data):
        if len(data) < 10:
            return False
        
        # Take the logarithm of the data
        log_data = [math.log(x) for x in data[-10:]]
        
        # Perform linear regression on log_data
        slope, _, _, _, _ = linregress(range(len(log_data)), log_data)
        
        # Check if the slope is negative
        return slope < 0
    
    def simulate(self, dates, dev_cut, profit_cut, coin, output) -> float: 
        profit = 1
        for date in dates:
            #LOG.info(f'{coin} {date}')
            df = self.import_individual_data(coin, date)
            times = df['time'].values
            vol = df['acc_trade_volume'].values
            devs = df['dev'].values
            price = df['trade_price'].values
            highs = df['high'].values
            status = 'sold'
            bought_coin = 'KRW'
            buying_price = 0
            bought_time = 0
            
            buying_price = 0
            max_profit = 0
            
            profit_list = [1.0]
            
            acc_vol = []
            start_time = times[0]
            
            for idx, dev in enumerate(devs):
                time = times[idx]
                if self.step[coin] == 0:
                    self.step[coin] = self.return_step(price[idx])
                # if time-start_time > duration:
                #     start_time = time
                #     start_vol = df['acc_trade_volume'].values[idx]
                #     if acc_vol < 0:
                #         start_vol = df['acc_trade_volume'].values[idx]
                #     acc_vol.append(start_vol)
                    
                if status == 'bought':
                    time_diff = times[idx]-bought_time
                    profit_cut[coin] = self.update_target_profit(time_diff)
    
                if status != "sold" and buying_price != 0:
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    max_price= price[idx]
                    max_idx = idx
                    max_profit = max(inst_profit, max_profit)
                else:
                    inst_profit = 1
                    
                if idx > 5:
                    mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                    
                condition_1 = (dev>=dev_cut and status == 'sold' 
                               and idx>5 and price[idx] <= highs[idx]-10*self.step[coin])
                # if dev>dev_cut:
                #     print('HIIIII')
                #     print(self.status)
                #     print(idx)
                if condition_1 == True:
                    status = 'buying'
                    buying_price = price[idx]
                    self.transaction_times[f'{status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{status}_idx'].append(idx)
                
                # if dev>=dev_cut:
                #     self.transaction_times['bought_times'].append(times[idx])
                #     self.transaction_idx[date]['bought_idx'].append(idx)
                
                    
                if status == 'buying' and buying_price >= price[idx]:
                    status = 'bought'
                    self.transaction_times[f'{status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{status}_idx'].append(idx)
                    buying_price = price[idx]
                    bought_time = times[idx]
                    
                    
                if status == 'bought' and inst_profit <= 0.98:
                    status = 'selling'
                    selling_price = price[idx]
                    self.transaction_times[f'{status}_times'].append(times[idx])
                    self.transaction_idx[date][f'{status}_idx'].append(idx)
                    
                if status == 'bought' and inst_profit>=profit_cut[coin]:
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
                    profit_list.append(profit)
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
                profit_list.append(profit)
                max_profit = 0
                
        sharpe = self.compute_sharpe_ratio(profit_list)
        
        if output == 'sharpe':
            return sharpe
        else:
            return profit

            
    def update_profit(self, dev_cut, profit_cut, date, coin, data_type):
        profit = self.simulate(dev_cut, profit_cut, date, data_type, coin)
        LOG.info(f'{coin} PROFIT for {date}: {self.daily_profits[date]}')
    
    def individual_simulation(self, coin:str, dev_cut:dict, profit_cut:dict):
        vis = generate_plots()
        for idx, date in enumerate(self.dates):
            df = self.import_individual_data(coin, date)
            profit = self.simulate([self.dates[idx]], dev_cut[coin], profit_cut, coin, 'profit')
            vis
            self.daily_profits[date] = profit
            print(self.transaction_idx[date])
            if profit != 1:
                vis.plot_transactions(df, self.transaction_idx[date], date, coin)
            else:
                vis.just_plot(df)
            
            self.transaction_idx = {
            date : {f'{status}_idx': [] for status in self.statuses} for date in self.dates
            }
                
        vis.plot_profits(coin, self.daily_profits)
        
    def update_target_profit(self, time_diff):
        return 1.001/0.9995+0.0999*np.exp(-time_diff/10)
        #return (1.001/0.9995+0.01*np.exp(-time_diff/100))
    
    def simulate_all(self, dev_cut:dict, profit_cut:dict, dates:list):
        profits = {}
        status = 'sold'
        profit = 1
        buying_price = 0
        max_profit = 0
        for date in dates:
            df = self.import_all(date)
            times = df['time'].values
            vol = df['acc_trade_volume'].values
            devs = df['dev'].values
            price = df['trade_price'].values
            highs = df['high'].values
            coins = df['coin'].values
            trading_coins = set(dev_cut.keys())
            traded_coins = []
            bought_coin = 'KRW'
            
            for idx, dev in enumerate(devs):
                coin = coins[idx]
                # if self.step[coin] == 0:
                #     self.step[coin] = self.return_step(price[idx])
                if status != "sold" and buying_price != 0 and coin == bought_coin:
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    max_profit = max(inst_profit, max_profit)
                else:
                    inst_profit = 1
                    
                if status == 'bought':
                    time_diff = times[idx]-bought_time
                    profit_cut[coin] = self.update_target_profit(time_diff)
                    
                if idx > 5:
                    mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                    
                condition_1 = (coin in trading_coins and dev>dev_cut[coin]
                            and idx>5)
                # if dev>dev_cut:
                #     print('HIIIII')
                #     print(self.status)
                #     print(idx)
                if condition_1 == True:
                    if status == 'sold':
                        status = 'buying'
                        buying_price = price[idx]
                        bought_coin = coin
                    # else:
                    #     bought_time = times[idx]
                    #LOG.info(f'buying {coin} at price: {buying_price}')
                    
                if coin in trading_coins and dev>=dev_cut[coin] and status == 'bought':
                    traded_coins.append(coins[idx])
                    
                if status == 'buying' and buying_price >= price[idx]:
                    status = 'bought'
                    bought_coin = coin
                    bought_time = times[idx]
                    bought_price = price[idx]
                    traded_coins.append(coins[idx])
                    
                if status == 'bought' and inst_profit <= 0.99:
                    status = 'selling'
                    selling_price = price[idx]
                    
                if status == 'bought' and coin in trading_coins and inst_profit>=profit_cut[coin]:
                    status = 'selling'
                    selling_price = price[idx]
                    
                if status == 'selling' and selling_price <= price[idx]:
                    status = 'sold'
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    profit *= inst_profit
                    #LOG.info(f'selling {coin} at price: {price[idx]} profit: {profit}')
                    max_profit = 0
                    buying_price = 0
                    
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
                filtered_df = df[df['coin'] == bought_coin]
                final_price = filtered_df['trade_price'].values[-1]
                inst_profit = (self.transaction*final_price/buying_price)
                profit *= inst_profit
                max_profit = 0
            traded_coins = set(traded_coins)
            LOG.info(f'{date} overall_profit: {profit}')
            profits[date] = profit
        return profits, traded_coins
    
    def simulate_using_avg(self, dates, trading_coins):
        profits = {}
        profit = 1
        status = 'sold'
        buying_price = 0
        bought_coin = 'KRW'
        for date in dates:
            df = self.import_all(date)
            times = df['time'].values
            vol = df['acc_trade_volume'].values
            devs = df['dev'].values
            price = df['trade_price'].values
            highs = df['high'].values
            coins = df['coin'].values
            traded_coins = []
            max_profit = 0
            
            upbit_coins = pyupbit.get_tickers("KRW")
            dev_cash = {coin: [] for coin in upbit_coins}
            avg_dev = {coin: 0 for coin in upbit_coins}
            profit_cut = {coin: 1.1 for coin in upbit_coins}
            
            for idx, dev in enumerate(devs):
                coin = coins[idx]
                if status != "sold" and buying_price != 0 and coin == bought_coin:
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    max_profit = max(inst_profit, max_profit)
                else:
                    inst_profit = 1
                    
                if status == 'bought':
                    time_diff = times[idx]-bought_time
                    profit_cut[coin] = self.update_target_profit(time_diff)
                if coin in trading_coins:
                    dev_cash[coin].append(dev)
                    if len(dev_cash[coin]) >= 1000:
                        subarray = np.abs(np.array(dev_cash[coin]))
                        avg_dev[coin] = np.mean(subarray)
                        #print(f'{coin} {avg_dev[coin]} {dev}')
                        dev_cash[coin].pop(0)
                    
                condition_1 = (coin in trading_coins and avg_dev[coin] != 0 and
                            dev>2e2*avg_dev[coin] and price[idx]<=0.99*highs[idx])
                # if dev>dev_cut:
                #     print('HIIIII')
                #     print(self.status)
                #     print(idx)
                if condition_1 == True:
                    if status == 'sold':
                        status = 'buying'
                        buying_price = price[idx]
                        bought_coin = coin
                        #print(status)
                    # else:
                    #     bought_time = times[idx]
                    #LOG.info(f'buying {coin} at price: {buying_price}')
                    
                if (coin in trading_coins and avg_dev[coin] != 0 and
                            dev>2e2*avg_dev[coin] and status == 'bought'):
                    traded_coins.append(coins[idx])
                    
                if status == 'buying' and buying_price >= price[idx]:
                    status = 'bought'
                    bought_coin = coin
                    bought_time = times[idx]
                    bought_price = price[idx]
                    traded_coins.append(coins[idx])
                    
                if status == 'bought' and inst_profit <= 0.97:
                    status = 'selling'
                    selling_price = price[idx]
                    
                if status == 'bought' and coin in trading_coins and inst_profit>=profit_cut[coin]:
                    status = 'selling'
                    selling_price = price[idx]
                    
                if status == 'selling' and selling_price <= price[idx]:
                    status = 'sold'
                    inst_profit = (self.transaction*price[idx]/buying_price)
                    profit *= inst_profit
                    #LOG.info(f'selling {coin} at price: {price[idx]} profit: {profit}')
                    max_profit = 0
                    buying_price = 0
                    
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
                filtered_df = df[df['coin'] == bought_coin]
                final_price = filtered_df['trade_price'].values[-1]
                inst_profit = (self.transaction*final_price/buying_price)
                profit *= inst_profit
                max_profit = 0
            LOG.info(f'date: {date} overall_profit: {profit}')
            profits[date] = profit
            
        return profits, traded_coins
        
            
    
    
        
            
        
    
    