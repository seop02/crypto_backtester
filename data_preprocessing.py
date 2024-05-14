import pandas as pd
import numpy as np
from backtest import data_path
import os
import pyupbit

def merge_data(coins, dates, trials):
    for date in dates:
        if date == '2024-02-20-2':
            date1  = date[0:10]
        for coin in coins:
            print(f'working on {coin}')
            original_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv'
            if os.path.exists(original_path):
                df = pd.read_csv(original_path, index_col=0)
                for trial in trials:
                    print(f'loading for {trial}')
                    additional_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume_{int(trial)}.csv'
                    if not os.path.exists(additional_path):
                        print('moving on to next one')
                    else:
                        print(f'adding {trial}')
                        df1 = pd.read_csv(additional_path, index_col=0)
                        df = pd.concat([df, df1], ignore_index=True)
                        df.to_csv(f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv')
                        os.remove(additional_path)
                        break
                
def merge_data2(dates, trials):
    for date in dates:
        original_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        for trial in trials:
            if os.path.exists(original_path):
                print(f'loading for {trial}')
                additional_path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                df1 = pd.read_csv(additional_path, index_col=0)
                df = pd.concat([df, df1], ignore_index=True)
                #os.remove(additional_path)
            else:
                path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                df = pd.read_csv(path, index_col=0)
                df.to_csv(original_path)
                #os.remove(path)
            next_path = path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)+1}.csv'
            if not os.path.exists(next_path):
                df.drop_duplicates(inplace=True)
                df.sort_values(by='traded_time', ascending=True, inplace=True)
                df.to_csv(original_path)
                break
                #os.remove(original_path)

def merge_data3(date, coins):
    target_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
    if not os.path.exists(f'{data_path}/ticker/{date}'):
        os.mkdir(f'{data_path}/ticker/{date}')
    for coin in coins:
        print(f'loading {coin}')
        file_path = f'{data_path}/acc/{date}/{date[0:10]}_{coin}_upbit_volume.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            n = len(df['time'].values)
            df['coin'] = [coin for _ in range(n)]
            if coin == coins[0]:
                df.to_csv(target_path)
                target_df = pd.read_csv(target_path, index_col=0)
            else:
                target_df = pd.concat([target_df, df], ignore_index=True)
    print('done loading, modifying data.....')
    target_df.drop_duplicates(inplace=True)
    target_df.sort_values(by='time', ascending=True, inplace=True)
    target_df.to_csv(target_path)
    print(target_df.head(10))
            
        
if __name__=='__main__':
    coins = pyupbit.get_tickers(fiat="KRW")
    #coins = ['KRW-BTC', 'KRW-DOGE', 'KRW-NEO', 'KRW-ONG', 'KRW-ONT', 'KRW-SHIB', 'KRW-XRP']
    dates = [
            '2024-05-13'
            ]
    
    trials = list(range(10000))
    merge_data3('2024-04-26', coins)
    #merge_dev(coins, dates, trials)
    # merge_data2(dates, trials)
    # date = dates[-1]
    # original_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
    # df = pd.read_csv(original_path, index_col=0)

    # df = df.rename(
    #     columns={'traded_time': 'time',
    #              'traded_price': 'trade_price',
    #              'acc_trade_vol': 'acc_trade_volume'})
    # df.to_csv(original_path)
    # print(len(df['time']))