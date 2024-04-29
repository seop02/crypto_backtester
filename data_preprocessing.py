import pandas as pd
import numpy as np
from backtest import data_path
import os
import pyupbit

def merge_data(coins, dates, trials):
    for date in dates:
        for coin in coins:
            print(f'working on {coin}')
            original_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume_0.csv'
            if os.path.exists(original_path):
                df = pd.read_csv(original_path, index_col=0)
                df.to_csv(f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume.csv')
                for trial in trials:
                    print(f'loading for {trial}')
                    additional_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume_{int(trial)}.csv'
                    if not os.path.exists(additional_path):
                        print('moving on to next one')
                        break
                    else:
                        print(f'adding {trial}')
                        df1 = pd.read_csv(additional_path, index_col=0)
                        df = pd.concat([df, df1], ignore_index=True)
                        df.to_csv(f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume.csv')
                        os.remove(additional_path)
                os.remove(original_path)
                
def merge_data2(coins, dates, trials):
    for date in dates:
        if date == '2024-02-20-2':
            date1  = date[0:10]
        for coin in coins:
            print(f'working on {coin}')
            original_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv'
            if os.path.exists(original_path):
                df = pd.read_csv(original_path, index_col=0)
                for trial in trials:
                    #print(f'loading for {trial}')
                    additional_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume_{int(trial)}.csv'
                    if not os.path.exists(additional_path):
                        #print('moving on to next one')
                        pass
                    else:
                        #print(f'adding {trial}')
                        df1 = pd.read_csv(additional_path, index_col=0)
                        df = pd.concat([df, df1], ignore_index=True)
                        df.to_csv(f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv')
                        os.remove(additional_path)
                        break
                #os.remove(original_path)
                
def merge_dev(coins, dates, trials):
    for date in dates:
        for coin in coins:
            print(f'working on {coin}')
            original_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_dev.csv'
            if os.path.exists(original_path):
                df = pd.read_csv(original_path, index_col=0)
                for trial in trials:
                    print(f'loading for {trial}')
                    additional_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_dev_{int(trial)}.csv'
                    if not os.path.exists(additional_path):
                        print('moving on to next one')
                    else:
                        print(f'adding {trial}')
                        df1 = pd.read_csv(additional_path, index_col=0)
                        df = pd.concat([df, df1], ignore_index=True)
                        df.to_csv(f'{data_path}/acc/{date}/{date}_{coin}_upbit_dev.csv')
                        os.remove(additional_path)
                        break
                    
def merge_all(coins, dates):
    combined_path = f'{data_path}/combined'
    if not os.path.exists(combined_path):
        os.mkdir(combined_path)
    for coin in coins:
        for date in dates:
            if date == '2024-01-31-1':
                file_path = f'{data_path}/acc/{date}/{date[:-2]}_{coin}_upbit_volume.csv'
            else:
                file_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                if date == dates[0]:
                    df_final = df
                    df_final.to_csv(f'{combined_path}/{dates[0]}_{dates[-1]}_{coin}.csv')
                else:
                    df_final = pd.concat([df_final, df], ignore_index=True)
                    df_final.to_csv(f'{combined_path}/{dates[0]}_{dates[-1]}_{coin}.csv')
                    
                
        
if __name__=='__main__':
    coins = pyupbit.get_tickers(fiat="KRW")
    coins = ['KRW-BTC', 'KRW-DOGE', 'KRW-NEO', 'KRW-ONG', 'KRW-ONT', 'KRW-SHIB', 'KRW-XRP']
    dates = [
            '2024-04-11'
            #'2024-04-03-1'
            ]
    
    trials = list(range(1,5000))
    coin = coins[0]

    #merge_dev(coins, dates, trials)
    merge_data2(coins, dates, trials)
    for date in dates:
        if date == '2024-02-20-2':
            date1  = date[0:10]
        for coin in coins:
            print(f'working on {coin}')
            original_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv'
            if os.path.exists(original_path):
                df = pd.read_csv(original_path, index_col=0)
                df.drop_duplicates(inplace=True)
                df.sort_values(by='time', ascending=True, inplace=True)
                df.to_csv(original_path)
    #merge_all(coins, dates)