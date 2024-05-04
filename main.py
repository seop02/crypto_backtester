import os
import sys
from backtest.simulator import backtrader
from backtest.optimizer import find_best_dev
from backtest.visualizer import generate_plots
import logging
import pyupbit

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=='__main__':
    dates = [
            # '2024-04-06', '2024-04-07',
            # '2024-04-09', '2024-04-10',
            '2024-04-11', '2024-04-26'
            #'2024-05-03'
            ]
    dev_cut = {
        'KRW-BTC': 1.8e-10,
        'KRW-GLM': 5.42e-05
    }
    
    profit_cut = {
        'KRW-BTC' : 1.03,
        'KRW-GLM' : 1.2
    }
    coins = list(dev_cut.keys())
    dict = {'2024-04-11': {'buying_idx': [], 'bought_idx': [], 'selling_idx': [], 'sold_idx': []}}
    sim = backtrader(coins, dates)
    for coin in coins:
        sim.individual_simulation(coin, dev_cut, profit_cut)
    # for coin in coins:
    #     opt = find_best_dev(coins, dates)
    #     best_dev_cut, best_profit_cut, best_profit = opt.optimize_dev(coin, dates[0])
    #     LOG.info(f'best dev: {best_dev_cut} best profit_cut: {best_profit_cut} best profit: {best_profit}')
    #     dev_cut[coin] = float(best_dev_cut)

    LOG.info(f'final dev_cut: {dev_cut}')