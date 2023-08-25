import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def preprocessing(scale, duration, cutoff):
    df = pd.read_csv(f'raw_10_{duration}.csv', index_col=0)
    excluding_columns = [
        'last', 'symbol', 'timestamp', 'datetime', 'bid', 
        'bidVolume', 'ask', 'askVolume', 'info', 'previousClose'
        ]
    
    step = int(scale/10)
    columns = [col for col in df.columns if col not in excluding_columns]
    x_raw = df.loc[0::step, columns]
    
    y_raw = df.loc[0::step, 'last'].values

    LOG.info(len(y_raw))

    y = []
    index = []
    for idx, price in enumerate(y_raw):
        if idx < len(y_raw)-1:
            change = (y_raw[idx+1]-price)/price
            if change > cutoff:
                label = 1
                y.append(label)

            elif change < -cutoff:
                label = 0
                y.append(label)
            
            else:
                index.append(idx)
  

    X = x_raw.values
    X = X[:-1]
    X = np.delete(X, index, axis=0)
    

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)

    X = np.array(X)
    y = np.array(y)
    return X, y

def one_class(scale, duration, cutoff):
    df = pd.read_csv(f'raw_10_{duration}.csv', index_col=0)
    excluding_columns = [
        'last', 'symbol', 'timestamp', 'datetime', 'bid', 
        'bidVolume', 'ask', 'askVolume', 'info', 'previousClose'
        ]
    
    step = int(scale/10)
    columns = [col for col in df.columns if col not in excluding_columns]
    x_raw = df.loc[0::step, columns]
    
    y_raw = df.loc[0::step, 'last'].values



    y = []
    for idx, price in enumerate(y_raw):
        if idx < len(y_raw)-1:
            change = (y_raw[idx+1]-price)/price
            if change > cutoff:
                label = -1
                y.append(label)

            elif change < -cutoff:
                label = -1
                y.append(label)
            
            else:
                label = 1
                y.append(label)
  

    X = x_raw.values
    X = X[:-1]
    
    # pca = PCA(n_components='mle', svd_solver='full')
    # pca.fit(X)
    # X = pca.transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X= np.array(X)
    return X, y