from backtest import analyze
from backtest.simulator import backtrader
from backtest.visualizer import generate_plots
import pyupbit
import pandas as pd
import numpy as np
from backtest import data_path, months

if __name__ == '__main__':
    finder = analyze.analyzer()
    vis = generate_plots()
    coin = 'KRW-ETC'
    dates = ['2024-04-11', '2024-04-26']
    trial = 0
    date = '2024-03'

    coins = pyupbit.get_tickers('KRW')
    #df = pd.read_parquet(f'{data_path}/ticker/{date}/upbit_volume.parquet')
    #df = df[df['coin'] == coin]
    # df = pd.read_csv(f'{data_path}/ticker/{date}/upbit_volume.csv', index_col=0)
    # df = df[df['coin'] == 'KRW-GLM']
    # vis.just_plot(df)
    # sim = backtrader(coins, dates)
    # dev = finder.abs_dev(df, 0.001)

