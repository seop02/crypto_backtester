import os
import sys
from backtest.simulator import backtrader
import pyupbit

if __name__=='__main__':
    dates = [
            # '2024-04-06', '2024-04-07',
            # '2024-04-09', '2024-04-10',
            '2024-04-11'
            ]
    dev_cut = {
        'KRW-BTC': 1.5e-10, 'KRW-ETH': 3e-9, 'KRW-ETC': 3.5e-6, 'KRW-ANKR': 0.00015, 'KRW-STORJ': 1e-6, 
    }
    avg_ratio = {
        'KRW-PUNDIX': 0.4e10, 'KRW-ADA':0.1e10
    }
    coins = list(dev_cut.keys())
    #coins = ['KRW-ETH']
    dict = {'2024-04-11': {'buying_idx': [], 'bought_idx': [], 'selling_idx': [], 'sold_idx': []}}
   
    sim = backtrader(coins, dates)
    sim.run_simulation(coins, dev_cut, 1.1, True)
