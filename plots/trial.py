import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=='__main__':
    data = np.load('orderbook_data_2.npy')
    
    X_raw = data
    X_raw = np.array(X_raw)
    length = X_raw.shape[0]
    X = np.delete(X_raw, [0,2], axis = 1)
    LOG.info(X[0])
    
    asks = X[:,0]
    bids = X[:,1]

    signal = asks - bids
    
    LOG.info(signal)

    price = []
    diff = []
    for idx in range(length):
        mid = (X_raw[idx][0][0] + X_raw[idx][2][0])/2
        price.append(mid)

    price = np.array(price)
    length = X.shape[0]

    time = np.linspace(0, length, length)

    fig: Figure = plt.figure(figsize =(20, 20))
    ax: Axes = fig.subplots()

    plt.plot(time, price)
    

    for idx, value in enumerate(signal):
      if value > 15:
        plt.scatter([time[idx]], [price[idx]], color = 'red')
      elif value < -15:
        plt.scatter([time[idx]], [price[idx]], color = 'blue')
      else:
        plt.scatter([time[idx]], [price[idx]], color = 'orange')

    plt.show()