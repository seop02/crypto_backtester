import os
import sys
from backtest.simulator import backtrader
from backtest.optimizer import find_best_dev
from backtest.visualizer import generate_plots
import logging
import pyupbit

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=='__main__':
    dates = [
            # '2024-04-06', '2024-04-07',
            # '2024-04-09', '2024-04-10',
            '2024-04-11', '2024-04-26',
            # '2024-05-03',
            # '2024-05-05'
            ]
    dev_cut = {'KRW-BTC': 1.8676075103041464e-10, 'KRW-GLM': 5.767264285795995e-05, 'KRW-ETH': 3.2096144610948675e-09, 'KRW-MTL': 9.374124632441e-06, 'KRW-XEM': 0.0008681642683769, 'KRW-LSK': 7.749928271933168e-06, 'KRW-STEEM': 0.0006271310315149, 'KRW-STORJ': 3.388945081117706e-05, 'KRW-ADA': 3.3649442174453224e-05, 'KRW-POWR': 6.7988391237141e-05, 'KRW-BTG': 2.1144173209672362e-07, 'KRW-EOS': 3.15556907988985e-05, 'KRW-SC': 0.0019826631641073, 'KRW-ONT': 4.472224659817241e-05, 'KRW-LOOM': 0.0002244270472308, 'KRW-BCH': 1.3412306228860553e-08, 'KRW-HIFI': 1.3267995384765625e-05, 'KRW-ONG': 4.911211822162517e-05, 'KRW-ELF': 3.464767178430198e-05, 'KRW-KNC': 3.9518187929749626e-05, 'KRW-THETA': 1.2623948240589864e-05, 'KRW-QKC': 0.0049990522581302, 'KRW-MOC': 0.0003074135774294, 'KRW-TFUEL': 5.6045706788286165e-05, 'KRW-ANKR': 0.0002105943367505, 'KRW-HBAR': 0.0001439573626826, 'KRW-MLK': 6.134307432118259e-05, 'KRW-STPT': 0.0004438318308202, 'KRW-CHZ': 0.0001053291096488, 'KRW-DKA': 0.001002373008413, 'KRW-HIVE': 3.875236085347862e-05, 'KRW-KAVA': 3.227671259164868e-05, 'KRW-AHT': 0.0171882509934369, 'KRW-LINK': 6.1559806886462e-07, 'KRW-XTZ': 2.5699949878701864e-05, 'KRW-CRO': 0.0001779648817787, 'KRW-SXP': 4.063566767091502e-05, 'KRW-HUNT': 4.497033876713867e-05, 'KRW-DOT': 2.2762178574192684e-06, 'KRW-SAND': 1.70444147748447e-05, 'KRW-PUNDIX': 9.7451673654245e-06, 'KRW-FLOW': 1.2878336750757102e-05, 'KRW-AXS': 2.474640536439949e-06, 'KRW-XEC': 0.2072190704331792, 'KRW-SOL': 3.3099947801338165e-08, 'KRW-AAVE': 3.2347770135234173e-07, 'KRW-1INCH': 4.880635483491921e-05, 'KRW-ALGO': 6.53036533456064e-05, 'KRW-NEAR': 5.840784263221978e-07, 'KRW-BLUR': 4.130038971873764e-05, 'KRW-ASTR': 0.0003582161060844}

    coins = list(dev_cut.keys())
    profit_cut = {coin: 1.2 for coin in coins}
    coins = pyupbit.get_tickers(fiat='KRW')
    dict = {'2024-04-11': {'buying_idx': [], 'bought_idx': [], 'selling_idx': [], 'sold_idx': []}}
    sim = backtrader(coins, dates)
    vis = generate_plots()
    df = sim.individual_simulation('KRW-LINK', dev_cut, profit_cut)
    # vis.plot_ticker(['KRW-GLM', 'KRW-BTG'], dates[-1])
    # for coin in coins:
    #     sim.individual_simulation(coin, dev_cut, profit_cut)
    
    #profit = sim.simulate_all(dates[-1], dev_cut, profit_cut)
    coins = ['KRW-LINK', 'KRW-ANKR', 'KRW-DOGE', 'KRW-XRP', 'KRW-TFUEL']
    # for coin in coins:
    #     LOG.info(f'optimizing for {coin}')
    #     opt = find_best_dev(coins, dates)
    #     best_dev_cut, best_profit = opt.optimize_dev(coin, dates[0])
        
    #     if best_profit > 1:
    #         LOG.info(f'best dev: {best_dev_cut} best profit: {best_profit}')
    #         dev_cut[coin] = float(best_dev_cut[0])

    #         LOG.info(f'final dev_cut: {dev_cut}')