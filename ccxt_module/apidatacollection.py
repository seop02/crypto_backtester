import ccxt #installed through pip install ccxt
import pprint
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def apidata(title, upbitcu, currency_binance, currency_upbit, duration):
    
    with open("ccxt_module/api_upbit.txt") as fu:
        lines_upbit = fu.readlines()
        api_key_upbit = lines_upbit[0].strip()
        secret_upbit  = lines_upbit[1].strip()

    with open("ccxt_module/api.txt") as f:
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

    upbit = ccxt.upbit(config={
        'apiKey': api_key_upbit, 
        'secret': secret_upbit,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })

    result_binance = []
    result_upbit = []
    i = 0
    while i < duration:
        btc_binance = binance.fetch_ticker(currency_binance)
        result_binance.append(btc_binance)
        df_binance = pd.DataFrame(data=result_binance)
        df_binance.to_csv(f'apidata_{title}_{upbitcu}_binance.csv')

        btc_upbit = upbit.fetch_ticker(currency_upbit)
        result_upbit.append(btc_upbit)
        df_upbit = pd.DataFrame(data=result_upbit)
        df_upbit.to_csv(f'apidata_{title}_{upbitcu}_upbit.csv')
        LOG.info(f'{i}')
        i += 1
