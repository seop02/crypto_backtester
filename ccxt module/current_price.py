import ccxt
import time
from datetime import datetime, timedelta
import pandas as pd

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
time_delta = timedelta(seconds=10)
end = now+10*time_delta
t = now
result = []
while t < end:
    btc = binance.fetch_ticker(symbol)
    result.append(btc)
    t += time_delta
   
df = pd.DataFrame(data=result)
df.to_csv('trial.csv')


