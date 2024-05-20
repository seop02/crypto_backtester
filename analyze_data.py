from backtest import analyze
from backtest.simulator import backtrader
import pyupbit
from backtest import data_path

if __name__ == '__main__':
    finder = analyze.analyzer()
    coin = 'KRW-BTC'
    date = '2024-03'
    trial = 0

    coins = pyupbit.get_tickers('KRW')
    #dev = finder.find_dev(date, coin, 2e2, 1)
    sim = backtrader(coins, [date])
    path = f'{data_path}/ticker/{date}/upbit_volume_{trial}.csv'
    profit, traded_coins = sim.simulate_using_avg(path, coins)