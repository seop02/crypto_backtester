import ccxt #installed through pip install ccxt
import pprint
import pandas as pd

#Access to the current market data
binance = ccxt.binance()
btc = binance.fetch_ticker("BTC/USDT")
pprint.pprint(btc) #print current price of BTC

#Access to the past market data
#minute by minute data
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT")
dfm = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
dfm['datetime'] = pd.to_datetime(dfm['datetime'], unit='ms')
dfm.set_index('datetime', inplace=True)
print(dfm)

#daily data
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", "1d") #add '1d' from minute by minute data
dfd = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
dfd['datetime'] = pd.to_datetime(dfd['datetime'], unit='ms')
dfd.set_index('datetime', inplace=True)
print(dfd)

#Note that the btc_ohlcv gives 500 rows by default(past 500 minutes, or past 500 days)
#Specify the amount of data we want via:
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe='1d', limit=10) # adding limit to fetch_ohlcv

df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
print(df)

def market_data(scale, duration):
    if scale == 'day':
        btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", "1d", limit=duration)
        df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
    elif scale == 'second':
        btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", limit=duration)
        df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        
    return df