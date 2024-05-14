from backtest.simulator import backtrader
from scipy.optimize import minimize
from backtest import data_path
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class find_best_dev(backtrader):
    date = '2024-04-11'
    dates = ['2024-04,11']
    coin = 'KRW-ANKR'
    profit_cut = {}
    def objective_function(self, x):
        dev_cut = x
        df = self.import_individual_data(self.coin, self.date)
        current_profit = self.simulate(self.dates, dev_cut, self.profit_cut, self.coin)
        #LOG.info(f'current profit: {current_profit}')
        return np.exp(-current_profit)
    
    def optimize_dev(self, coin, dates:list, dev_cut:dict):
        total_profit = 1.0
        self.coin = coin
        self.dates = dates
        date = dates[0]
        self.profit_cut[coin] = 1.2
        
        default_date = datetime.strptime('2024-04-28', '%Y-%m-%d')
        input_date = datetime.strptime(date, '%Y-%m-%d')
        
        if input_date>default_date:
            file_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        else:
            file_path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            if input_date>default_date:
                df = df[df['coin']==coin]
            if coin in set(dev_cut.keys()):
                init_dev = dev_cut[coin]
            else:
                init_dev = np.max(df['dev'].values)
            initial_guess = [init_dev]  # Initial guess for dev_cut and profit_cut
            bounds = [(1e-13, 2)]  # Bounds for dev_cut and profit_cut
            
            result = minimize(self.objective_function, initial_guess, bounds=bounds, method='Nelder-Mead')
            
            best_dev_cut = result.x
            best_profit = -np.log(result.fun)  # Convert back to positive profit
        
            return best_dev_cut, best_profit
        else:
            return 1, 1
            