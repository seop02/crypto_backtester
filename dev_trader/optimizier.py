from optimize_parameters import dev_trader
from scipy.optimize import minimize
from dev_trader import data_path
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

def simulate(coin, dates, dev_cut, avg_cut, profit_cut):
    file_paths = [f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv' for date in dates]
    transaction = 0.9995**2
    statuses = ['buying', 'bought', 'selling', 'sold']
    transaction_times = {f'{status}_times': [] for status in statuses}
    
    def return_profit1(dev_cut, avg_cut, profit_cut):
        profit = 1
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                times = df['time'].values
                vol = df['acc_trade_volume'].values
                devs = df['dev'].values
                price = df['trade_price'].values
                highs = df['high'].values
                ask_bid = df['ask_bid'].values
                
                
                status = 'sold'
                
                buying_price = 0
                max_profit = 0
                
                for idx, dev in enumerate(devs):
                    if status != "SOLD" and buying_price != 0:
                        inst_profit = (transaction*price[idx]/buying_price)-1
                        max_price= price[idx]
                        max_idx = idx
                        max_profit = max(inst_profit, max_profit)
                    else:
                        inst_profit = 0
                        
                    if idx > 5:
                        mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                        
                    condition_1 = dev>dev_cut and status == 'sold'
                    
                    if condition_1 == True and dev/mean_dev > avg_cut:
                        status = 'buying'
                        buying_price = price[idx]
                        transaction_times[f'{status}_times'].append(times[idx])
                        
                    if status == 'buying' and buying_price >= price[idx]:
                        status = 'bought'
                        transaction_times[f'{status}_times'].append(times[idx])
                        
                    if status == 'bought' and inst_profit <= -0.01:
                        status = 'selling'
                        selling_price = price[idx]
                        transaction_times[f'{status}_times'].append(times[idx])
                        
                    if status == 'bought' and  max_profit>0.01 and inst_profit<max_profit/3:
                        status = 'selling'
                        selling_price = price[idx]
                        transaction_times[f'{status}_times'].append(times[idx])
                    
                    if status == 'bought' and inst_profit>profit_cut:
                        status = 'selling'
                        selling_price = price[idx]
                        transaction_times[f'{status}_times'].append(times[idx])
                    
                    if status == 'selling' and selling_price <= price[idx]:
                        status = 'sold'
                        transaction_times[f'{status}_times'].append(times[idx])
                        inst_profit = (transaction*price[idx]/buying_price)
                        profit *= inst_profit
                        max_profit = 0
                        
                    if status == 'buying':
                        time_diff = times[idx]-transaction_times[f'{status}_times'][-1]
                        if time_diff > 30:
                            status = 'sold'
                    
                    if status == 'selling':
                        time_diff = times[idx]-transaction_times[f'{status}_times'][-1]
                        if time_diff > 30:
                            status = 'bought'
            
            else:
                print(f'{file_path} does not exists!!!')
        return -profit
    return return_profit(dev_cut, avg_cut, profit_cut)

def return_profit(x):
    dev_cut, avg_cut, profit_cut = x[0], x[1], x[2]
    profit = 1
    file_paths = [f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv' for date in dates]
    transaction = 0.9995**2
    statuses = ['buying', 'bought', 'selling', 'sold']
    transaction_times = {f'{status}_times': [] for status in statuses}
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            times = df['time'].values
            vol = df['acc_trade_volume'].values
            devs = df['dev'].values
            price = df['trade_price'].values
            highs = df['high'].values
            ask_bid = df['ask_bid'].values
            
            
            status = 'sold'
            
            buying_price = 0
            max_profit = 0
            
            for idx, dev in enumerate(devs):
                if status != "SOLD" and buying_price != 0:
                    inst_profit = (transaction*price[idx]/buying_price)-1
                    max_price= price[idx]
                    max_idx = idx
                    max_profit = max(inst_profit, max_profit)
                else:
                    inst_profit = 0
                    
                if idx > 5:
                    mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
                else:
                    mean_dev = 1
                    
                condition_1 = dev>abs(dev_cut) and status == 'sold' and idx>5
                
                if condition_1 == True and mean_dev >0:
                    if dev/mean_dev > avg_cut:
                        status = 'buying'
                        buying_price = price[idx]
                        transaction_times[f'{status}_times'].append(times[idx])
                    
                if status == 'buying' and buying_price >= price[idx]:
                    status = 'bought'
                    transaction_times[f'{status}_times'].append(times[idx])
                    
                if status == 'bought' and inst_profit <= -0.01:
                    status = 'selling'
                    selling_price = price[idx]
                    transaction_times[f'{status}_times'].append(times[idx])
                    
                if status == 'bought' and  max_profit>0.01 and inst_profit<max_profit/3:
                    status = 'selling'
                    selling_price = price[idx]
                    transaction_times[f'{status}_times'].append(times[idx])
                
                if status == 'bought' and inst_profit>profit_cut:
                    status = 'selling'
                    selling_price = price[idx]
                    transaction_times[f'{status}_times'].append(times[idx])
                
                if status == 'selling' and selling_price <= price[idx]:
                    status = 'sold'
                    transaction_times[f'{status}_times'].append(times[idx])
                    inst_profit = (transaction*price[idx]/buying_price)
                    profit *= inst_profit
                    max_profit = 0
                    
                if status == 'buying':
                    time_diff = times[idx]-transaction_times[f'{status}_times'][-1]
                    if time_diff > 30:
                        status = 'sold'
                
                if status == 'selling':
                    time_diff = times[idx]-transaction_times[f'{status}_times'][-1]
                    if time_diff > 30:
                        status = 'bought'
        
        else:
            print(f'{file_path} does not exists!!!')
    return -profit

if __name__ == "__main__":
    coin = 'KRW-PUNDIX'
    dev_cut = {
        'KRW-BTC': 5e-14, 'KRW-ETH': 6e-8, 'KRW-ETC': 3.5e-6, 'KRW-ANKR': 0.00015, 'KRW-STORJ': 1e-6, 
        'KRW-MBL': 0.04, 'KRW-MED': 0.001, 'KRW-DKA': 0.024, 'KRW-BTG': 1e-6, 'KRW-GLM': 2e-7, 
        'KRW-NEAR': 4e-6, 'KRW-JST': 0.0007, 'KRW-TFUEL': 0.03, 'KRW-QTUM': 0.003,
        'KRW-ID': 0.0002, 'KRW-CELO': 0.0005, 'KRW-POWR': 0.0004, 'KRW-PDA': 0.0003, 'KRW-IOST': 0.001,
        'KRW-ZRX': 1e-6, 'KRW-LSK': 1e-6, 'KRW-HIFI': 1e-6, 'KRW-SOL': 1e-6, 'KRW-CTC': 2e-6,
        'KRW-MASK': 1e-6, 'KRW-AVAX': 1e-6, 'KRW-APT': 1e-6, 'KRW-STRAX': 1e-6, 'KRW-MVL': 0.0004,
        'KRW-GRS': 9e-6, 'KRW-PUNDIX': 1e-6, 'KRW-HUNT': 1e-6, 'KRW-SUI': 1e-6, 'KRW-CVC': 1e-6, 'KRW-T': 1e-4,
        'KRW-TON': 1e-6, 'KRW-WAXP': 4e-5, 'KRW-HBAR': 1e-5, 'KRW-MTL': 1e-4, 'KRW-META': 1e-4, 'KRW-XEM': 1e-4,
        'KRW-HPO': 2.5e-6, 'KRW-ADA': 3.5e-6
        }
    coins = list(dev_cut.keys())
    dates = ['2024-03-22', '2024-03-23',
            '2024-03-25',
            '2024-03-26', 
            '2024-03-28', '2024-03-29', '2024-03-30',
            '2024-03-31',
            '2024-04-01',
            '2024-04-02',
            '2024-04-03',
            '2024-04-04',
            '2024-04-06', '2024-04-07',
            '2024-04-09', '2024-04-10']
    optimal = {}
    coins = ['KRW-BTC']
    for coin in coins:
        print('============================================================')
        print(f'coin: {coin}')
        trader = dev_trader(coin, dates)
        bnds = ((1e-15, 1e-3), (1e7, 1e13), (0.01, 0.5))
        profit = -1
        # while profit >= -1:
        #     x1 = np.random.uniform(1e-15, 1e-3)
        #     x2 = np.random.uniform(1e7, 1e13)
        #     x3 = np.random.uniform(0.01, 0.5)
        x0 = [dev_cut[coin], 1e7, 0.05]
        profit = trader.update_profit(x0[0], x0[1], x0[2])
        print(f'initial profit: {profit}')
        bnds = ((1e-15, 1e-3), (1e7, 1e13), (0.01, 0.5))
        optimized_parameters = minimize(return_profit, x0=x0, bounds=bnds, method='Nelder-Mead')
        print(f'optimizied parameters: {optimized_parameters.x}')
        res = optimized_parameters.x
        #res = [8.75422634e-05,  4.00000000e+09,  9.99535213e-02]
        profit = trader.update_profit(res[0], res[1], res[2])
        print(f'optimized profit: {profit}')
        optimal[coin] = res
        print('============================================================')
    print(optimal)
