import pandas as pd
import numpy as np
from __init__ import data_path
import os
import pyupbit
                
def merge_data(dates, trials):
    for date in dates:
        original_path = f'{data_path}/ticker/{date}/upbit_volume.csv'
        for trial in trials:
            if os.path.exists(original_path):
                #df = pd.read_csv(original_path, index_col=0)
                additional_path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                print(f'adding {trial}')
                df1 = pd.read_csv(additional_path, index_col=0)
                df = pd.concat([df, df1], ignore_index=True) 
                df.drop_duplicates(inplace=True)
                df.sort_values(by='traded_time', ascending=True, inplace=True)
                os.remove(additional_path)
            else:
                file_path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)}.csv'
                df = pd.read_csv(file_path, index_col=0)
                df.to_csv(original_path)
                os.remove(file_path)
            next_path = f'{data_path}/ticker/{date}/upbit_volume_{int(trial)+1}.csv'
            if not os.path.exists(next_path):
                df.to_csv(original_path)
                break
            
if __name__=='__main__':
    
    dates = [
            '2024-05-03'
            ]
    trials = list(range(5000))
    original_path = f'{data_path}/ticker/{dates[0]}/upbit_volume.csv'

    df = pd.read_csv(original_path, index_col=0)
    print(df.head())
    df = df.rename(columns={'acc_trade_vol':'acc_trade_volume'})
    df.to_csv(original_path)

    