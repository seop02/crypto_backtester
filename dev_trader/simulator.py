import pyupbit
import pandas as pd
import numpy as np
from dev_trader import data_path
import math
import os
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class dev_trader():
    def __init__(self, coin, dates):
        self.coin = coin
        self.dates = dates
        self.file_paths = [f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv' for date in dates]
        self.transaction = 0.9995**2
        self.profit = 1
        statuses = ['buying', 'bought', 'selling', 'sold']
        self.transaction_times = {f'{status}_times': [] for status in statuses}
        
        
    def return_step(self, price):
        step = {}
        if price<0.01:
            step[self.coin] = 0.000001
        elif 0.01<=price<0.1:
            step[self.coin] = 0.00001
        elif 1<price<10:
            step[self.coin] = 0.001
        elif 10<=price<100:
            step[self.coin] = 0.01
        elif 100<=price<1000:
            step[self.coin] = 0.1
        elif 1000<=price<10000:
            step[self.coin] = 1
        elif 10000<=price<100000:
            step[self.coin] = 10
        elif 100000<=price<1000000:
            step[self.coin] = 50
        elif 1000000<=price:
            step[self.coin] = 1000
        #LOG.info(f'updating step of {coin} to {step[coin]}')
        return step[self.coin]

    def round_sigfigs(self, num, sig_figs):
        if num != 0:
            return round(num, -int(math.floor(math.log10(abs(num)))) + (sig_figs - 1))
        else:
            return 0.0
    
    def simulate(self, dev_cut, avg_cut, profit_cut):
        for file_path in self.file_paths:
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
                        
                    condition_1 = dev>dev_cut and self.status == 'sold' and idx>5 and mean_dev != 0
                    
                    if condition_1 == True and dev/mean_dev > avg_cut:
                        self.status = 'buying'
                        buying_price = price[idx]
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                        
                    if self.status == 'buying' and buying_price >= price[idx]:
                        self.status = 'bought'
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                        
                    if self.status == 'bought' and inst_profit <= -0.01:
                        self.status = 'selling'
                        selling_price = price[idx]
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                        
                    if self.status == 'bought' and  max_profit>0.01 and inst_profit<max_profit/3:
                        self.status = 'selling'
                        selling_price = price[idx]
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                    
                    if self.status == 'bought' and inst_profit>profit_cut:
                        self.status = 'selling'
                        selling_price = price[idx]
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                    
                    if self.status == 'selling' and selling_price <= price[idx]:
                        self.status = 'sold'
                        self.transaction_times[f'{self.status}_times'].append(times[idx])
                        inst_profit = (self.transaction*price[idx]/buying_price)
                        self.profit *= inst_profit
                        max_profit = 0
                        
                    if self.status == 'buying':
                        time_diff = times[idx]-self.transaction_times[f'{self.status}_times'][-1]
                        if time_diff > 30:
                            self.status = 'sold'
                    
                    if self.status == 'selling':
                        time_diff = times[idx]-self.transaction_times[f'{self.status}_times'][-1]
                        if time_diff > 30:
                            self.status = 'bought'
            
            else:
                print(f'{file_path} does not exists!!!')
            
    def update_profit(self, dev_cut, avg_cut, profit_cut):
        self.simulate(dev_cut, avg_cut, profit_cut)
        LOG.info(f'PROFIT: {self.profit}')
        return self.profit
        
            
        
    
    