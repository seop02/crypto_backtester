from market_data import market_data
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    scale = 20
    duration = 1000
    current_datetime = datetime.now()
    symbol = 'BTC/USDT'
    
    market_data(scale, duration, symbol)
    
    #print(df['last'])