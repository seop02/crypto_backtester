import sys
sys.path.append("./")
from ccxt_module import orders
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=='__main__':
    currency = 'XRP/USDT'
    balance = orders.current_balance()
    usd_balance = balance['USDT']
    xrp_balance = balance['XRP']
    LOG.info(usd_balance)
    


    