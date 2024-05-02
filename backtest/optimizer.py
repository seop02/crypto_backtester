from backtest.simulator import backtrader
from scipy.optimize import minimize
from backtest import data_path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class find_best_dev(backtrader):
    date = '2024-04-11'
    coin = 'KRW-ANKR'
    def objective_function(self, x):
        dev_cut, profit_cut = x
        current_profit = self.simulate(dev_cut, profit_cut, self.date, self.coin)
        LOG.info(f'current profit: {current_profit}')
        return np.exp(-current_profit)
    
    def optimize_dev(self, coin, date):
        total_profit = 1.0
        self.coin = coin
        self.date = date
        
        file_path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
        df = pd.read_csv(file_path, index_col=0)
        init_dev = np.max(df['dev'].values)
        initial_guess = [init_dev, 1.2]  # Initial guess for dev_cut and profit_cut
        bounds = [(1e-12, 1e-3), (1, 1.5)]  # Bounds for dev_cut and profit_cut
        
        result = minimize(self.objective_function, initial_guess, bounds=bounds, method='Nelder-Mead')
        
        best_dev_cut, best_profit_cut = result.x
        best_profit = -result.fun  # Convert back to positive profit
        
        return best_dev_cut, best_profit_cut, best_profit
            