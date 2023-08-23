import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


if __name__ == '__main__':
    df = pd.read_csv('raw_10_8640.csv')
    df2 = pd.read_csv('trial_data.csv', index_col=0)
    df3 = pd.read_csv('actual.csv', index_col=0)
   
    estimated = df2['pred'].values
    actual = df3['actual'].values

    price = df['last'].values
    time = np.linspace(0, 81390, 8139)
    time2 = np.linspace(65040, 81390, 272)

    figure, ax = plt.subplots(1, 2)

    ax[0].plot(time, price, color  = 'black')
    ax[1].plot(time, price, color  = 'black')
    for idx, value in enumerate(actual):
      if value == 1:
         ax[0].scatter([time2[idx]], [price[6*idx+6504]], color = 'red')
      elif value == 2:
         ax[0].scatter([time2[idx]], [price[6*idx+6504]], color = 'blue')
      elif value == 0:
         ax[0].scatter([time2[idx]], [price[6*idx+6504]], color = 'orange')
   
    for idx, value in enumerate(estimated):
         if value == 1:
            ax[1].scatter([time2[idx]], [price[6*idx+6504]], color = 'red', marker='x')
         elif value == 2:
            ax[1].scatter([time2[idx]], [price[6*idx+6504]], color = 'blue', marker='x')
         elif value == 0:
            ax[1].scatter([time2[idx]], [price[6*idx+6504]], color = 'orange', marker='x')
      
    plt.show()
    
