import pandas as pd
import numpy as np
from backtest import data_path
import os
import pyupbit
import pyarrow

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
        if os.path.exists(original_path):
            df = pd.read_csv(original_path, index_col=0)
        for trial in trials:
            if os.path.exists(original_path):
                print(f'loading for {trial}')
                additional_path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                if os.path.exists(additional_path):
                    df1 = pd.read_csv(additional_path, index_col=0)
                    df = pd.concat([df, df1], ignore_index=True)
                    #os.remove(additional_path)
            else:
                path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                df = pd.read_csv(path, index_col=0)
                df.to_csv(original_path)
                #os.remove(path)
            next_path = path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)+1}.csv'
            #if not os.path.exists(next_path):
                
        df.drop_duplicates(inplace=True)
        df.sort_values(by='traded_time', ascending=True, inplace=True)
        df = df.rename(
            columns={'traded_time': 'time',
                    'traded_price': 'trade_price',
                    'acc_trade_vol': 'acc_trade_volume'})
        df.to_csv(original_path)
           
                #os.remove(original_path)

def merge_data3(date, coins):
    target_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
    if not os.path.exists(f'{data_path}/ticker/{date}'):
        os.mkdir(f'{data_path}/ticker/{date}')
    for coin in coins:
        #print(f'loading {coin}')
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
    
def combine_dates(dates, label):
    final_path = f'{data_path}/ticker/{dates[0][0:7]}/upbit_volume.csv'
    if not os.path.exists(f'{data_path}/ticker/{dates[0][0:7]}'):
        os.mkdir(f'{data_path}/ticker/{dates[0][0:7]}')
    for date in dates:
        print(f'loading {date}')
        file_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        df = pd.read_csv(file_path, index_col=0)
        print(df.shape[1])
        if date == dates[0]:
            target_df = df
            #target_df.to_parquet(final_path, engine='pyarrow')
            target_df.to_csv(final_path)
        else:
            target_df = pd.concat([target_df, df], ignore_index=True)
    print(target_df.head(10))
    df['time'] = df['time'].astype(str)
    #target_df.to_parquet(final_path, engine='pyarrow')
    target_df.to_csv(final_path)
    
    
            
            
        
if __name__=='__main__':
    coins = pyupbit.get_tickers(fiat="KRW")
    
    directory_path = f"{data_path}/ticker"
    
    months = {
    "feb" : ['2024-02-21-2', '2024-02-22', '2024-02-22-1', '2024-02-22-2', '2024-02-22-3', '2024-02-23', '2024-02-23-1', '2024-02-24', '2024-02-25', '2024-02-26', '2024-02-27', '2024-02-27-1', '2024-02-29'],
    "mar" : ['2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-06', '2024-03-07', '2024-03-08', '2024-03-09', '2024-03-11', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-16', '2024-03-17', '2024-03-18', '2024-03-18-1', '2024-03-19', '2024-03-20', '2024-03-20-1', '2024-03-22', '2024-03-23', '2024-03-25', '2024-03-25-1', '2024-03-26', '2024-03-27', '2024-03-28', '2024-03-29', '2024-03-30', '2024-03-31'],
    "apr" : ['2024-04-01', '2024-04-01-1', '2024-04-02', '2024-04-03', '2024-04-03-1', '2024-04-04', '2024-04-05', '2024-04-06', '2024-04-07', '2024-04-11', '2024-04-26'],
    "may" : ['2024-05-02', '2024-05-03', '2024-05-05', '2024-05-06', '2024-05-10', '2024-05-11', '2024-05-12',  '2024-05-15', '2024-05-16', '2024-05-17', '2024-05-18', '2024-05-19', '2024-05-20']
    }


    combine_dates(months["apr"], 0)
    #dates = ['2024-0']
    #coins = ['KRW-BTC', 'KRW-DOGE', 'KRW-NEO', 'KRW-ONG', 'KRW-ONT', 'KRW-SHIB', 'KRW-XRP']
    # trials = list(range(272,10000))
    #dates = ['2024-02-29']
    # for date in dates:
    #     print(date)
    #     if len(date)<=11:
    #         print(date)
    #         merge_data3(date, coins)
    # #merge_dev(coins, dates, trials)
    # dates = [
    #          '2024-05-18'
    #          ]
    # trials = list(range(3000))
    # merge_data2(dates, trials)
    # date = dates[-1]
    # original_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
    # df = pd.read_csv(original_path, index_col=0)

    
    # # print(len(df['time']))