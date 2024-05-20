import os
import sys

data_path = f"{os.path.dirname(__file__)}/../../../../../Volumes/ssd/data"
error_path = f"{os.path.dirname(__file__)}/../error"
volume_path = f"{os.path.dirname(__file__)}/../volume_data"
balance_path = f"{os.path.dirname(__file__)}/../balance"

months = {
    "feb" : ['2024-02-21-2', '2024-02-22', '2024-02-22-1', '2024-02-22-2', '2024-02-22-3', '2024-02-23', '2024-02-23-1', '2024-02-24', '2024-02-25', '2024-02-26', '2024-02-27', '2024-02-27-1', '2024-02-29'],
    "mar" : ['2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-06', '2024-03-07', '2024-03-08', '2024-03-09', '2024-03-11', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-16', '2024-03-17', '2024-03-18', '2024-03-18-1', '2024-03-19', '2024-03-20', '2024-03-20-1', '2024-03-22', '2024-03-23', '2024-03-25', '2024-03-25-1', '2024-03-26', '2024-03-27', '2024-03-28', '2024-03-29', '2024-03-30', '2024-03-31'],
    "apr" : ['2024-04-01', '2024-04-01-1', '2024-04-02', '2024-04-03', '2024-04-03-1', '2024-04-04', '2024-04-05', '2024-04-06', '2024-04-07', '2024-04-11', '2024-04-26'],
    "may" : ['2024-05-02', '2024-05-06', '2024-05-10', '2024-05-11', '2024-05-12', '2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17']
}


# dev_cut = {}
#     coins = pyupbit.get_tickers(fiat='KRW')
#     coins.remove('KRW-AKT')
#     coins.remove('KRW-ZETA')
    
#     opt_profits = {coin: 0 for coin in coins}
#     for coin in coins:
#         LOG.info(f'optimizing for {coin}')
#         opt = find_best_dev(coins, dates)
#         best_dev_cut, best_profit = opt.optimize_dev(coin, dates, dev_cut)
        
#         if best_profit > 1:
#             LOG.info(f'best dev: {best_dev_cut} best profit: {best_profit}')
#             new_dev_cut[coin] = float(best_dev_cut[0])
#             opt_profits[coin] = best_profit

#         LOG.info(f'final dev_cut: {new_dev_cut}')
        
#     sorted_dict = dict(sorted(opt_profits.items(), key=lambda x: x[1], reverse=True))
#     LOG.info(f'profits: {sorted_dict}')