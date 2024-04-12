import pandas as pd
import numpy as np
import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyupbit
from dev_trader import data_path
import logging
import math

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def flatten_nested_list(nested_list):
    flattened_list = [item for sublist in nested_list for item in sublist]
    return flattened_list

def moving_average(prices, window_size):
    moving_averages = []
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    return moving_averages

def return_step(price):
    step = {}
    if price<0.01:
        step[coin] = 0.000001
    elif 0.01<=price<0.1:
        step[coin] = 0.00001
    elif 1<price<10:
        step[coin] = 0.001
    elif 10<=price<100:
        step[coin] = 0.01
    elif 100<=price<1000:
        step[coin] = 0.1
    elif 1000<=price<10000:
        step[coin] = 1
    elif 10000<=price<100000:
        step[coin] = 10
    elif 100000<=price<1000000:
        step[coin] = 50
    elif 1000000<=price:
        step[coin] = 1000
    #LOG.info(f'updating step of {coin} to {step[coin]}')
    return step[coin]

def round_sigfigs(num, sig_figs):
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num)))) + (sig_figs - 1))
    else:
        return 0.0

def plot_derivative(coin, date, mode, duration, status, dev_cut, plot_ma):
    if duration == 'daily':
        #LOG.info(date[0:10])
        file_path = f'{data_path}/acc/{date}/{date[:10]}_{coin}_upbit_volume.csv'
    elif duration == 'combined':
        file_path = f'{data_path}/combined/2024-01-29_2024-02-06_{coin}.csv'
        
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        times = df['time'].values
        vol = df['acc_trade_volume'].values
        devs = df['dev'].values
        price = df['trade_price'].values
        highs = df['high'].values
        ask_bid = df['ask_bid'].values
        dev_new = []
        n = len(times)
        #devs = devs*price
        
        max_dev = 0
        max_price = 0
        min_price = 0
        bought = 0
        
        derivatives = [0]
        time_differences = []
        
        transaction = 0.9995**2
        b_idx = 0
        
        buying_times, buying_prices = [], []
        bought_times, bought_prices = [], []
        selling_times, selling_prices = [], []
        
        
        good_buys = []
        bad_buys = []
        
        bought_price = {}
        balance = 10000
        
        r_idx = 0
        good_data = []
        bad_data = []
        profits = []
        step = return_step(price[0])
        tot_profit = 1
        avg_devs = 1
        counter = 0
        max_profit = 0
        duration = 100
        diff = 300
        derivatives1 = []
        derivatives2 = []
        ma100 = np.zeros(n-101)
        ma1000 = np.zeros(n-1001)
        if n>=10001:
            ma10000 = np.zeros(n-10001)
        profit_cut = -0.01
        h_idx = 0
        
        high_count = 0

        time_diff = 0
        b_idx = 0
        max_idx = 0

        mean_dev = 0
        #devs = devs/vol
        
        for idx, dev in enumerate(devs):
            gradient = 1
            if max_dev != 0:  
                ratio = dev/max_dev
            else:
                ratio = -1
            # if idx>1:
            #     dev *= (times[idx]-times[idx-1])
            # if ask_bid[idx] == 'ASK':
            #     sign = -1
            # elif ask_bid[idx] == 'BID':
            #     sign = 1
            if idx>=1:
                
                abs_devs = (vol[idx]-vol[idx-1])
                if abs_devs < 0:
                    abs_devs = 0

                if ask_bid[idx] == 'ASK':
                    sign = -1
                elif ask_bid[idx] == 'BID':
                    sign = 1

                abs_devs *= sign
                #LOG.info(abs_devs)
                derivatives.append(abs_devs)
            if idx > diff:
                time_differences.append(1/(times[idx]-times[idx-diff]))
                
            # #dev = sigdev
            # dev_new.append(dev)
            if idx > 1000:
                mean_dev = np.mean(np.square(np.array(devs[idx-5:idx])))
            
            max_price = max(price[idx], max_price)
            
            price_diff = (max_price-price[idx])/price[idx]
            
            if status != "SOLD":
                profit = (transaction*price[idx]/buying_price)-1
                max_price= price[idx]
                max_idx = idx
                max_profit = max(profit, max_profit)
            else:
                profit = 0
            if idx>100 and plot_ma == True:
                window = np.array(price[idx-99:idx+1])
                average = np.sum(window)/100
                #LOG.info(len(window))
                ma100[idx-101] = average
            if idx>1000 and plot_ma == True:
                window = np.array(price[idx-999:idx+1])
                average = np.sum(window) / 1001
                ma1000[idx-1001] = average
            if idx>10000 and n >= 10001 and plot_ma == True:
                window = np.array(price[idx-9999:idx+1])
                average = np.sum(window) / 10001
                ma10000[idx-10001] = average
                        
           
            
            condition_2 = dev > 50*max_dev and times[idx]-times[0]>1800 and (max_price-5*step>=price[idx]>=max_price-10*step) and status == "SOLD"

            condition_1 = idx >= 1 and devs[idx-1] != 0 and abs(dev) > 1e12*abs(devs[idx-1]) and status == 'SOLD' and price[idx] > highs[idx]-40*step#and times[idx]-times[h_idx] > 500

            condition_1 = dev>dev_cut[coin] and status == 'SOLD' and idx>10000
            
            if condition_1 == True:
                time_diff = times[idx]-times[0]
                
               
                # sublist = np.abs(np.array(devs[idx-500:idx-2]))
                # raw_sublist = devs[idx-500:idx-2]
                # std = np.std(sublist/np.linalg.norm(sublist))
                # sub_min = np.min(raw_sublist)
                # sub_max = np.max(sublist)
                # normalized_diff = (sub_max+abs(sub_min))/(sub_max)
                # price_list = np.array(price[idx-1000:idx-1])/np.mean(price[idx-1000:idx-1])
                # price_std = np.std(price_list)
                # max_price = np.max(price_list)
                # min_price = np.min(price_list)
                # volume_list = np.array(devs[idx-1000:idx])
                # avg_vol = np.mean(np.abs(volume_list))
                # max_vol = max(abs(np.mean(volume_list)), np.max(volume_list))

                #LOG.info(f'buying with diff: {normalized_diff} price_std: {price_std} max_vol: {max_vol} current_dev: {dev}')
                
                if dev/mean_dev>0.4e10:
                    #LOG.info(f'sending buying order for {coin} at time: {times[idx]} at {price[idx]} dev: {dev} max_dev: {max_dev} ratio: {dev/max_dev} std: {normalized_diff} price_std: {price_std}')
                #if 0.09 < std < 0.11:
                #if True:
                    #mean_dev = np.mean(np.array(devs[:idx]))
                    status = "BUY"
                    LOG.info(f'max_ratio: {dev/max_dev} price: {price[idx]} mean: {mean_dev} mean_ratio: {dev/mean_dev}')

                    # price_list = np.array(price[idx-500:idx-1])
                    # price_std = np.std(price_list)
                    # max_price = np.max(price_list)
                    # min_price = np.min(price_list)

                    # LOG.info(f'range: {max_price-min_price}')
                    
                    #LOG.info(f'mean: {avg_vol} current: {dev} max: {np.max(volume_list)}')
                    
                    if coin == 'KRW-BTC':
                        buying_price = round_sigfigs(price[idx] - 0*step , 5)
                    else:
                        buying_price = round_sigfigs(price[idx] - 0*step , 4)
                    target_price = (1.01*buying_price)/transaction
                    if coin == 'KRW-BTC':
                        selling_price = round_sigfigs(target_price, 5)
                    else:
                        selling_price = round_sigfigs(target_price, 4)
                    b_idx = idx
                    bought_coin = coin
                    
                    bought_std = dev/mean_dev
                    #bought_std = (highs[idx]-price[idx])/price[idx]
                    bought_price_diff = (highs[idx]-price[idx])/price[idx]
                    bought_ratio = dev/max_dev
                    #bought_gradient = (price[idx]-price[idx-100])/(price[idx-100]*(times[idx]-times[idx-100]))
                    
                    
                    amount = 0.9995*balance/buying_price
                    buying_times.append(times[idx])
                    buying_prices.append(price[idx])
                    

            if status == "BUY" and price[idx] <= buying_price:
                #LOG.info(f'buying {coin} at price: {price[idx]} dev: {bought_ratio}')
                status = "BOUGHT"
                b_idx = idx
                min_price = price[idx]
                bought_times.append(times[idx])
                bought_prices.append(price[idx])
                
            if status == 'BOUGHT' and idx>1:
                if highs[idx] == highs[idx-1]:
                    high_count += 1
            if price[idx] > highs[idx]-5*step:
                h_idx = idx
                    #LOG.info(f'{coin} min profit while holding: {min_profit}')
                    
            if status == 'BOUGHT' and profit <= -0.01:
                #LOG.info(f'bad selling {coin}')
                #LOG.info(f'bad selling for {coin} with max_pro: {max_profit} dev: {dev} max_dev: {max_dev} bought_time: {times[b_idx]} std: {bought_std}  high_count: {high_count}')
                status = "SOLD"
                LOG.info(f'max_profit: {max_profit}')
                profit_cut = -0.005
                profit =  (transaction*price[idx]/buying_price)
                tot_profit *= profit
                profits.append(profit)
                selling_times.append(times[idx])
                selling_prices.append(price[idx])
                max_profit = 0
                high_count = 0
                time_diff = times[idx] - times[b_idx]
                LOG.info(f'max_idx: {max_idx}')
                max_idx = 0
                
                bad_data.append([bought_std, bought_ratio, time_diff])
                bad_buys.append(price[b_idx-20:])
            if status == 'BOUGHT' and max_profit>0.01 and profit<max_profit/3:
                #LOG.info(f'bad selling {coin}')
                #LOG.info(f'selling for {coin} with max_pro: {max_profit} dev: {dev} max_dev: {max_dev} bought_time: {times[b_idx]} std: {bought_std}  high_count: {high_count}')
                status = "SOLD"
                #LOG.info(f'max_profit: {max_profit}')
                profit_cut = -0.005
                profit =  (transaction*price[idx]/buying_price)
                tot_profit *= profit
                profits.append(profit)
                selling_times.append(times[idx])
                selling_prices.append(price[idx])
                max_profit = 0
                high_count = 0
                time_diff = times[idx] - times[b_idx]
                #LOG.info(f'max_idx: {max_idx}')
                max_idx = 0
                
                bad_data.append([bought_std, bought_ratio, time_diff])
                bad_buys.append(price[b_idx-20:])
            
            if status == 'BOUGHT' and profit>0.05:
                LOG.info(f'good selling {coin} at price: {price[idx]}')
                #LOG.info(f'good selling for {coin} with max_pro: {max_profit} dev: {dev} max_dev: {max_dev} bought_time: {times[b_idx]} std: {bought_std} high_count: {high_count}')
                status = "SOLD"
                profit_cut = -0.01
                time_diff = times[idx] - times[b_idx]
                profit =  (transaction*price[idx]/buying_price)
                tot_profit *= profit
                profits.append(profit)
                selling_times.append(times[idx])
                selling_prices.append(price[idx])
                good_data.append([bought_std, bought_ratio, time_diff])
                max_profit = 0
                high_count = 0
            
                good_buys.append(price[b_idx-20:])
                   
            if status == 'BUY':
                if times[idx]-times[b_idx] > 250:
                #LOG.info(f'cancelling buying order {status}')
                    status = 'SOLD'
            # if status == 'BOUGHT':
            #     if times[idx]-times[b_idx] > 40000:
            #     #LOG.info(f'cancelling selling order {status} {profit}')
            #         #LOG.info(f'kind of bad selling for {coin} with max_pro: {max_profit} dev: {dev} max_dev: {max_dev} bought_time: {times[b_idx]} std: {bought_std} high: {high_count}')
            #         status = 'SOLD'
            #         profit_cut = -0.01
            #         profit =  (transaction*price[idx]/buying_price)
            #         tot_profit *= profit
            #         profits.append(profit)
            #         selling_times.append(times[idx])
            #         selling_prices.append(price[idx])
                    
                    bad_data.append([bought_std, bought_ratio, bought_price_diff])
            if dev > max_dev:
                #LOG.info(f'updating max_dev from {max_dev} to {dev}')
                max_dev = max(max_dev, dev)
            

            if 502>idx-r_idx>500:
                max_dev *= 0.1
                r_idx = idx
            # if idx == max_idx and max_idx != 0:
            #     diff1 = (price[idx]-price[idx-100])/(price[idx-100]*(times[idx]-times[idx-100]))
            #     diff2 = (price[idx]-price[idx-1000])/(price[idx-100]*(times[idx]-times[idx-1000]))
            #     diff3 = (price[idx]-price[idx-5000])/(price[idx-100]*(times[idx]-times[idx-5000]))
                #LOG.info(f"diff100: {diff1} diff1000: {diff2} diff5000: {diff3}")
            
            
            # if times[idx]-times[0]>20000:
            #     avg_devs = 0
            #     counter += 1
            
            # avg_devs *= (idx-counter*duration)/((idx+1)-counter*duration)
            # avg_devs += dev/((idx+1)-counter*duration)
            
            # avg_devs *= (idx)/((idx+1))
            # avg_devs += dev/((idx+1))


        if status == 'BOUGHT':
            profit = (transaction*price[idx]/buying_price)
            #LOG.info(f'kind of bad selling for {coin} with max_pro: {max_profit} ratio: {bought_ratio} price_diff: {bought_price_diff} bought_time: {times[b_idx]} std: {bought_std}')
            tot_profit *= profit
            profit_cut = -0.01
            profits.append(profit)
            selling_times.append(times[idx])
            selling_prices.append(price[idx])
            bad_data.append([bought_std, bought_ratio, time_diff])
        #LOG.info(profits)
        #LOG.info([a - b for a, b in zip(bought_times, buying_times)])
        #LOG.info([a - b for a, b in zip(selling_times, bought_times)])

        if mode == 'show' and plot_ma == False:
            fig: Figure = plt.figure(figsize=(10,8))
            ax1, ax2, ax3 = fig.subplots(3)
            ax1.plot(times, price, color='black')
            ax2.plot(times, devs, color='black') 
            ax1.plot(times, highs, color='blue', linestyle='--')
            #ax1.scatter(bought_times, bought_prices, color='blue')
            ax1.scatter(bought_times, bought_prices, color='blue')
            ax1.scatter(selling_times, selling_prices, color='red')
            ax3.plot(times[diff+1:], time_differences, color='black')
            # ax1.scatter(signalt, signalp, color='green')
            # ax1.scatter(signalbt, signalbp, color='purple')
            plt.title(f'{date} {coin}')
            plt.show()

        elif mode == 'show' and plot_ma == True:
            fig: Figure = plt.figure(figsize=(10,8))
            ax1, ax2, ax3 = fig.subplots(3)
            ax1.plot(times, price, color='black')
            ax2.plot(times, devs, color='black') 
            ax1.plot(times[101:], ma100, label='100')
            ax1.plot(times[1001:], ma1000, label='1000')
            #ax1.plot(times[10001:], ma10000, label='10000')
            ax1.plot(times, highs, color='blue', linestyle='--')
            #ax1.scatter(bought_times, bought_prices, color='blue')
            ax1.scatter(bought_times, bought_prices, color='blue')
            ax1.scatter(selling_times, selling_prices, color='red')
            ax3.plot(times[diff:], time_differences, color='black')
            # ax1.scatter(signalt, signalp, color='green')
            # ax1.scatter(signalbt, signalbp, color='purple')
            ax1.legend()
            plt.title(f'{date} {coin}')
            plt.show()
        
        if len(buying_times) != 0:
            return bought_coin, tot_profit, buying_times, buying_prices, profits, good_data, bad_data, max_price, time_diff
        else:
            return 'no', tot_profit, [], [], 0, [], [], 0, 0
        
    else:
        tot_profit = 1
        LOG.info('path does not exist')
        return 'no', tot_profit, [], [], 0, [], [], 0, 0

def data_extract(coin, date, mode):
    file_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        times = df['time'].values
        devs = df['dev'].values
        price = df['trade_price'].values
        
        
        max_dev = 0
        max_price = 0
        bought = 0
        
        
        b_idx = 0
        
        good_buy = []
        bad_buy = []
        
        bought_price = {}
        profit = {}
        
        tot_profit = 1

        for idx, dev in enumerate(devs):
            max_price = max(price[idx], max_price)
            if bought != 0:
                profit = (price[idx]-price[b_idx])/price[b_idx]
            else:
                profit = 0
            
            if dev > 2*max_dev:
                if times[idx]-times[0]>1800 and price[idx]<0.999*max_price:
                    #print(f'buying {coin}!!')
                    
                    bought = 1
                    b_idx = idx
                    bought_coin = coin
                    bought_price[times[b_idx]] = price[idx]
                    profit[times[b_idx]] = 0
 
            # if dev > 0.4*max_dev:
            #     max_dev = max(0.9*max_dev, dev)
                
            if (profit > 0.004 or profit < -0.01) and bought == 1:
                bought = 0
                profit = 1+(price[idx]-price[b_idx])/price[b_idx]
                tot_profit *= profit
                sublist = devs[b_idx-9:b_idx+1]
                sublist = list(np.array(sublist) / np.linalg.norm(sublist))
                sublist1 = price[b_idx-9:b_idx+1]
                avg_p = sum(sublist1)/len(sublist1)
                sublist1 = sublist1-avg_p
                
                final_list = np.concatenate([sublist1, sublist])
                good_buy.append()
                
            if times[idx]-times[b_idx]>120 and bought == 1:
                bought = 0
                profit = 1+0.999*(price[idx]-price[b_idx])/price[b_idx]
                tot_profit *= profit
        if bought == 1:
            profit = 1+0.999*(price[idx]-price[b_idx])/price[b_idx]
            tot_profit *= profit

        if mode == 'show':
            fig: Figure = plt.figure(figsize=(10,7))
            ax1, ax2 = fig.subplots(2)
            ax1.plot(times, price, color='black')
            ax2.plot(times, devs, color='black') 
            ax1.scatter(buying_times, buying_prices, color='blue')
            plt.title(f'{date} {coin}')
            plt.show()
        if len(buying_times) != 0:
            return bought_coin, tot_profit, buying_times, buying_prices
        else:
            return 'no', tot_profit, [], []
        
    else:
        tot_profit = 1
        return 'no', tot_profit
    #LOG.info(f'data:{date} coin: {coin} profit: {tot_profit}')

def plot_simultaneously(date, coins, buying_times, buying_prices):
    fig: Figure = plt.figure(figsize=(10,7))
    for coin in coins:
        file_path = f'{data_path}/acc/{date}/{date}_{coin}_upbit_volume.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            prices = df['trade_price'].values
            scaled = prices/np.mean(prices)
            plt.plot(df['time'].values, scaled, label=coin)
            plt.scatter(buying_times[coin], buying_prices[coin]/np.mean(prices), color='blue')
    plt.legend()
    plt.show()
    
def calculate_profit(tot_profit, date, mode):
    new_profit = {key: value-1 for key, value in tot_profit.items() if value != 1}
    total = 1
    #plot_simultaneously(date, good_coins, buying_times, buying_prices)
    if len(new_profit) != 0 and mode == True:
        labels, values = zip(*new_profit.items())
        fig: Figure = plt.figure(figsize =(20, 5))
        plt.bar(labels, values, color='green', alpha=0.7)
        plt.title(f'Percentage profit for {date}')
        plt.show()
    else:
        LOG.info('NO TRADE!!')
        
    good_coins = set()
    bad_coins = set()
    
    for crypto, value in new_profit.items():
        total *= (1+value)
        #LOG.info(crypto)
        if value > 0:
            good_coins.add(crypto)
        elif value < -0.01:
            bad_coins.add(crypto)
                
    LOG.info(f'{date} overall_profit: {total}')
    return total, good_coins, bad_coins, len(new_profit)

if __name__=='__main__':
    dates = [
            # '2024-01-01', 
            # '2024-01-10', 
            # '2024-01-13', '2024-01-14', 
            # '2024-01-15', '2024-01-16',
            # '2024-01-17', '2024-01-18',
            # '2024-01-20', 
            # '2024-01-24'
            # '2024-01-25',
            # '2024-01-29',
            # '2024-01-30',
            # '2024-01-31',
            # '2024-01-31-1',
            # '2024-02-01',
            # '2024-02-02',
            # '2024-02-03',
            # '2024-02-05',
            # '2024-02-06',
            # '2024-02-08',
            # '2024-02-10',
            # '2024-02-11',
            # '2024-02-12',
            # '2024-02-13',
            # '2024-02-14',
            # '2024-02-15',
            # '2024-02-16',
            # '2024-02-16-1',
            # '2024-02-17',
            # '2024-02-18',
            # '2024-02-19',
            # '2024-02-20',
            # '2024-02-20-1',
            # '2024-02-20-2',
            #'2024-02-21-2', 
            # '2024-02-22', '2024-02-22-1', '2024-02-22-2',
            #'2024-02-22-3',
            # '2024-02-23', 
            # '2024-02-24',
            # '2024-02-25',
            # '2024-02-26',
            # '2024-02-27', '2024-02-27-1',
            # '2024-02-29',
            # '2024-03-01', 
            # '2024-03-02',
            # '2024-03-03',
            # '2024-03-04',
            # '2024-03-06',
            # '2024-03-07', '2024-03-08',
            # '2024-03-09', '2024-03-11',
            # '2024-03-12', '2024-03-13', '2024-03-14',
            # '2024-03-15',
            #'2024-03-16', 
            # '2024-03-17',
            # '2024-03-18',
            # '2024-03-19',
            # '2024-03-20',
            '2024-03-22', '2024-03-23',
            '2024-03-25',
            '2024-03-26', 
            '2024-03-28', '2024-03-29', '2024-03-30',
            '2024-03-31',
            '2024-04-01',
            '2024-04-02',
            '2024-04-03',
            '2024-04-04'
            '2024-04-06', '2024-04-07',
            '2024-04-09', '2024-04-10'
            ]
    
    
    
    #coins = ['KRW-BTC']
    #coins = ['KRW-SUI']#, 'KRW-ETH', 'KRW-ETC', 'KRW-XRP', 'KRW-ANKR', 'KRW-STORJ', 'KRW-STX', 'KRW-MBL']#, 'KRW-IQ', 'KRW-GRT', 'KRW-ANKR', 'KRW-BTC']
    total = 1
    dev_cut = {
        'KRW-BTC': 5e-12, 'KRW-ETH': 6e-8, 'KRW-ETC': 3.5e-6, 'KRW-ANKR': 0.00015, 'KRW-STORJ': 1e-6, 
        'KRW-MBL': 0.04, 'KRW-MED': 0.001, 'KRW-DKA': 0.024, 'KRW-BTG': 1e-6, 'KRW-GLM': 2e-7, 
        'KRW-NEAR': 4e-6, 'KRW-JST': 0.0007, 'KRW-TFUEL': 0.03, 'KRW-QTUM': 0.003,
        'KRW-ID': 0.0002, 'KRW-CELO': 0.0005, 'KRW-POWR': 0.0004, 'KRW-PDA': 0.0003, 'KRW-IOST': 0.001,
        'KRW-ZRX': 1e-6, 'KRW-LSK': 1e-6, 'KRW-HIFI': 1e-6, 'KRW-SOL': 1e-6, 'KRW-CTC': 2e-6,
        'KRW-MASK': 1e-6, 'KRW-AVAX': 1e-6, 'KRW-APT': 1e-6, 'KRW-STRAX': 1e-6, 'KRW-MVL': 0.0004,
        'KRW-GRS': 9e-6, 'KRW-PUNDIX': 1e-6, 'KRW-HUNT': 1e-6, 'KRW-SUI': 1e-6, 'KRW-CVC': 1e-6, 'KRW-T': 1e-4,
        'KRW-TON': 1e-6, 'KRW-WAXP': 4e-5, 'KRW-HBAR': 1e-5, 'KRW-MTL': 1e-4, 'KRW-META': 1e-4, 'KRW-XEM': 1e-4,
        'KRW-HPO': 2.5e-6, 'KRW-ADA': 3.5e-6
        }
    avg_ratio = {
        'KRW-PUNDIX': 0.4e10, 'KRW-ADA':0.1e10
    }
    
    
    coins = list(dev_cut.keys())
    coins = ['KRW-HPO']
    #coins = pyupbit.get_tickers(fiat="KRW")
    #coins = ['KRW-IOST', 'KRW-STRAX', 'KRW-MED', 'KRW-TON', 'KRW-T', 'KRW-APT', 'KRW-WAXP']
    #LOG.info(list(set(coins)-set(uiniverse)))  
    profit_list = []
    time_diffs = []
    traded_coins = []
    buying_times = []
    bought_times = {date: {coin: [] for coin in coins} for date in dates}
    bought_prices = {date: {coin: [] for coin in coins} for date in dates}
    good_datas = []
    bad_datas = []
    overall_good = set()
    overall_bad = set()
    for date in dates:
        tot_profit = {}
        good_coins = []
        daily = 1
        for coin in coins:
            status = 'SOLD'
            good_coin, tot_profit[coin], buying_time, buying_prices, profit, good_data, bad_data, max_price, time_diff  = plot_derivative(coin, date, 'show', 'daily', status, dev_cut, False)
            if len(good_data) != 0:
                status = 'SOLD'
                #good_coin, tot_profit[coin], buying_time, buying_prices, profit, bought_time, selling_time, max_price, a = plot_derivative(coin, date, 'show', 'daily', status, dev_cut, False)
                for data in good_data:
                        good_datas.append(data)
            if len(bad_data) != 0:
                status = 'SOLD'
                #good_coin, tot_profit[coin], buying_time, buying_prices, profit, bought_time, selling_time, max_price, a = plot_derivative(coin, date, 'show', 'daily', status, dev_cut, False)
                for data in bad_data:
                    bad_datas.append(data)

            
                
        new_bought = bought_times[date]
        filtered_bought = {key: value for key, value in new_bought.items() if len(value) != 0}
        
        sorted_dict = dict(sorted(filtered_bought.items(), key=lambda x: x[1][0]))
        LOG.info(sorted_dict)                
        daily_profit, daily_good, daily_bad, num_coins = calculate_profit(tot_profit, date, False)
        traded_coins.append(num_coins)
        overall_good = overall_good.union(daily_good)
        overall_bad = overall_bad.union(daily_bad)
        LOG.info(overall_good)
        LOG.info(overall_bad)
        total *= daily_profit
        profit_list.append(total)
        LOG.info(profit_list)
    
    LOG.info(f'TOTAL PROFIT: {total}')
    LOG.info(f'avg number of traded coins: {np.mean(np.array(traded_coins))}')
    fig: Figure = plt.figure(figsize =(20, 5))
    plt.plot(dates, profit_list)
    plt.show()
    
            
    #profits = flatten_nested_list(profits)
    #buying_times = flatten_nested_list(buying_times)
    #bad_datas = flatten_nested_list(bad_datas)
    #good_datas = flatten_nested_list(good_datas)
    
    #good_datas = [item[0] for item in good_data]
    #bad_datas = [item[0] for item in bad_data]
    
    # LOG.info(good_datas)
    # LOG.info(bad_datas)
    # y_1 = np.ones(len(good_datas))
    # y_2 = np.zeros(len(bad_datas))
    
    # X = np.concatenate([good_datas, bad_datas])
    # y = np.concatenate([y_1, y_2])
    
    # np.save(f'{data_path}/scalp_X_1.npy', X)
    # np.save(f'{data_path}/scalp_y_1.npy', y)
    LOG.info(good_datas)
    LOG.info(bad_datas)
    good_X = [point[0] for point in good_datas]
    good_Y = [point[1] for point in good_datas]
    good_Z = [point[2] for point in good_datas]
    good_n = len(good_Z)
    good_scale = np.ones((good_n))
    
    bad_X = [point[0] for point in bad_datas]
    bad_Y = [point[1] for point in bad_datas]
    bad_Z = [point[2] for point in bad_datas]
    bad_n = len(bad_Z)
    bad_scale = np.ones((bad_n))

    # fig = plt.subplots(figsize=(10, 7))
    # plt.scatter(good_scale, good_Z, color='blue')
    # plt.scatter(bad_scale, bad_Z, color='red')
    # plt.show()
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    axs[0, 0].scatter(good_X, good_Y, color='blue', label='Good')
    axs[0, 0].scatter(bad_X, bad_Y, color='red', label='Bad')
    axs[0, 0].set_xlabel('HIGH DIFF')
    axs[0, 0].set_ylabel('Ratio')
    axs[0, 0].set_title('Scatter plot of Gradient vs. Ratio')
    axs[0, 0].legend()

    axs[0, 1].scatter(good_X, good_Z, color='blue', label='Good')
    axs[0, 1].scatter(bad_X, bad_Z, color='red', label='Bad')
    axs[0, 1].set_xlabel('HIFH DIFF')
    axs[0, 1].set_ylabel('TIME DIFF')
    axs[0, 1].set_title('Scatter plot of Gradient vs. Price_diff')
    axs[0, 1].legend()

    axs[1, 0].scatter(good_Y, good_Z, color='blue', label='Good')
    axs[1, 0].scatter(bad_Y, bad_Z, color='red', label='Bad')
    axs[1, 0].set_xlabel('RATIO')
    axs[1, 0].set_ylabel('TIME DIFF')
    axs[1, 0].set_title('Scatter plot of Ratio vs. Price_diff')
    axs[1, 0].legend()

    axs[1, 1].axis('off')  # You may choose to leave this subplot empty

    plt.tight_layout()
    plt.show()

    