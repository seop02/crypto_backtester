from backtest.simulator import backtrader
from scipy.optimize import minimize
from backtest import data_path
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class find_best_dev(backtrader):
    date = '2024-04-11'
    coin = 'KRW-ANKR'
    profit_cut = 1.2
    def objective_function(self, x):
        dev_cut = x
        df = self.import_individual_data(self.coin, self.date, 'acc')
        current_profit = self.simulate(df, dev_cut, self.profit_cut, self.date)
        #LOG.info(f'current profit: {current_profit}')
        return np.exp(-current_profit)
    
    def optimize_dev(self, coin, date):
        total_profit = 1.0
        self.coin = coin
        self.date = date
        
        file_path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
        if os.path.exists(file_path):
            
            df = pd.read_csv(file_path, index_col=0)
            init_dev = np.max(df['dev'].values)
            initial_guess = [init_dev]  # Initial guess for dev_cut and profit_cut
            bounds = [(1e-12, 1)]  # Bounds for dev_cut and profit_cut
            
            result = minimize(self.objective_function, initial_guess, bounds=bounds, method='Nelder-Mead')
            
            best_dev_cut = result.x
            best_profit = -np.log(result.fun)  # Convert back to positive profit
        
            return best_dev_cut, best_profit
        else:
            return 1, 1
            