import ccxt #installed through pip install ccxt
import pprint
import pandas as pd
from datetime import datetime, timedelta
import time

#Access to the current market data
binance = ccxt.binance()
btc = binance.fetch_ticker("BTC/USDT")
pprint.pprint(btc) #print current price of BTC

# #Access to the past market data
# #minute by minute data
# binance = ccxt.binance()
# btc_ohlcv = binance.fetch_ohlcv("BTC/USDT")
# dfm = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
# dfm['datetime'] = pd.to_datetime(dfm['datetime'], unit='ms')
# dfm.set_index('datetime', inplace=True)
# print(dfm)

#daily data
# binance = ccxt.binance()
# btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", "1s") #add '1d' from minute by minute data
# dfd = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
# dfd['datetime'] = pd.to_datetime(dfd['datetime'], unit='ms')
# dfd.set_index('datetime', inplace=True)
# print(dfd)

#Note that the btc_ohlcv gives 500 rows by default(past 500 minutes, or past 500 days)
#Specify the amount of data we want via:
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe='1s', limit=8640) # adding limit to fetch_ohlcv

df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
df.to_csv('raw.csv')
print(df)

def market_data(scale, duration, symbol):
    
    with open("ccxt module/api.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        secret  = lines[1].strip()
    
    binance = ccxt.binance(config={
        'apiKey': api_key, 
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })

    symbol = "BTC/USDT"
    now = datetime.now()
    time_delta = timedelta(seconds=scale)
    end = now+duration*time_delta
    t = now
    result = []
    while t < end:
        btc = binance.fetch_ticker(symbol)
        result.append(btc)
        t += time_delta
        df = pd.DataFrame(data=result)
        df.to_csv(f'raw_{scale}_{duration}.csv')
        time.sleep(10)
