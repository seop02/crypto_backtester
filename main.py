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
        'KRW-BTC': 5e-12, 'KRW-ETH': 6e-8, 'KRW-ETC': 3.5e-6, 'KRW-ANKR': 0.00015, 'KRW-STORJ': 1e-6, 
        'KRW-MBL': 0.04, 'KRW-MED': 0.001, 'KRW-DKA': 0.024, 'KRW-BTG': 1e-6, 'KRW-GLM': 2e-7, 
        'KRW-NEAR': 4e-6, 'KRW-JST': 0.0007, 'KRW-TFUEL': 0.03, 'KRW-QTUM': 0.003,
        'KRW-ID': 0.0002, 'KRW-CELO': 0.0005, 'KRW-POWR': 0.0004, 'KRW-PDA': 0.0003, 'KRW-IOST': 0.001,
        'KRW-ZRX': 1e-6, 'KRW-LSK': 1e-6, 'KRW-HIFI': 1e-6, 'KRW-SOL': 1e-6, 'KRW-CTC': 2e-6,
        'KRW-MASK': 1e-6, 'KRW-AVAX': 1e-6, 'KRW-APT': 1e-6, 'KRW-STRAX': 1e-6, 'KRW-MVL': 0.0004,
        'KRW-GRS': 9e-6, 'KRW-PUNDIX': 1e-6, 'KRW-HUNT': 1e-6, 'KRW-SUI': 1e-6, 'KRW-CVC': 1e-6, 'KRW-T': 1e-4,
        'KRW-TON': 1e-6, 'KRW-WAXP': 4e-5, 'KRW-HBAR': 1e-5, 'KRW-MTL': 1e-4, 'KRW-META': 1e-4, 'KRW-XEM': 1e-4,
        'KRW-HPO': 2.5e-6, 'KRW-ADA': 3.5e-6
        }
    avg_ratio = {
        'KRW-PUNDIX': 0.4e10, 'KRW-ADA':0.1e10
    }
    coins = list(dev_cut.keys())
    
    sim = backtrader(coins, dates)
    sim.run_simulation(coins, dev_cut, 1.05, True)