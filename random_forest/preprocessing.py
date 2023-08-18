import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def preprocessing():
    df = pd.read_csv('raw.csv', index_col=0)
    excluding_columns = [
        'last', 'symbol', 'timestamp', 'datetime', 'bid', 
        'bidVolume', 'ask', 'askVolume', 'info', 'previousClose'
        ]
    columns = [col for col in df.columns if col not in excluding_columns]
    x_raw = df.loc[:, columns]
    
    y_raw = df['last'].values

    y = []
    index = []
    for idx, price in enumerate(y_raw):
        if idx < len(y_raw)-1:
            change = y_raw[idx+1]-price 
            if change > 0:
                label = 1
                y.append(label)

            elif change < 0:
                label = 0
                y.append(label)
            
            else:
                index.append(idx)
  
    
    
    X = x_raw.values
    X = X[:-1]
    X = np.delete(X, index, axis=0)
    LOG.info(X.shape)
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    LOG.info(X.shape)
    
    return X, y