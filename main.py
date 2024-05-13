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
        #     '2024-04-11', '2024-04-26',
            '2024-05-03',
            '2024-05-05',
            '2024-05-06',
            '2024-05-10',
            '2024-05-11',
            '2024-05-12'
            ]
    dev_cut = {}

    coins = list(dev_cut.keys())
    profit_cut = {coin: 1.2 for coin in coins}
    coins = pyupbit.get_tickers(fiat='KRW')
    dict = {'2024-04-11': {'buying_idx': [], 'bought_idx': [], 'selling_idx': [], 'sold_idx': []}}
    sim = backtrader(coins, dates)
    vis = generate_plots()
    
#     profits = []
#     for date in dates:
#         profit, traded_coins = sim.simulate_all(date, dev_cut, profit_cut)
#         profits.append(profit)
#     LOG.info(f'profits: {profits}')
    coins = pyupbit.get_tickers(fiat='KRW')
    for coin in coins:
        LOG.info(f'optimizing for {coin}')
        opt = find_best_dev(coins, dates)
        best_dev_cut, best_profit = opt.optimize_dev(coin, dates)
        
        if best_profit > 1.02:
            LOG.info(f'best dev: {best_dev_cut} best profit: {best_profit}')
            dev_cut[coin] = float(best_dev_cut[0])

        LOG.info(f'final dev_cut: {dev_cut}')