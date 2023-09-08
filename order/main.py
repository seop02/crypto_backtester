import sys
sys.path.append("./")
from ccxt_module import orders, price_data
import logging
from decimal import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=='__main__':
    currency = 'XRP/USDT'
    balance = orders.current_balance()
    usd_balance = balance['USDT']
    xrp_balance = balance['XRP']
    bit_balance = balance['BTC']
    bit_amount = bit_balance['free']
    LOG.info(bit_amount)
    
    bit = 'BTC/USDT'
    current_ask, current_bid = price_data.current_prices(bit)
    current_price = (current_ask[0][0] + current_bid[0][0])/2
    amount = usd_balance['free']/current_price
   
    #buy = orders.insert_order(bit, amount, current_price)
    
    balance = orders.current_balance()
    usd_balance = balance['USDT']
    xrp_balance = balance['XRP']
    bit_balance = balance['BTC']
    bit_amount = bit_balance['free']
    LOG.info(usd_balance)
    LOG.info(bit_amount)
    
    current_price = (current_ask[0][0] + current_bid[0][0])/2
    amount = bit_amount
    #sell = orders.sell_order(bit, amount, current_price)
    usd_balance = balance['USDT']
    xrp_balance = balance['XRP']
    bit_balance = balance['BTC']
    bit_amount = bit_balance['free']
    LOG.info(usd_balance)
    LOG.info(bit_amount)
    


    