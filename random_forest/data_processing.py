import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

if __name__ == '__main__':
    # df = pd.read_csv('raw.csv', index_col=0)
    # excluding_columns = [
    #     'last', 'symbol', 'timestamp', 'datetime', 'bid', 
    #     'bidVolume', 'ask', 'askVolume', 'info', 'previousClose'
    #     ]
    # columns = [col for col in df.columns if col not in excluding_columns]
    # x_raw = df.loc[:, columns]
    
    # y_raw = df['last'].values
    # print(y_raw)
    # y = []
    # for idx, price in enumerate(y_raw):
    #     if idx < len(y_raw)-1:
    #         change = y_raw[idx+1]-price 
    #         if change > 0:
    #             label = 1
    #         else:
    #             label = 0
    #         y.append(label)
    # #print(y)
    # X = x_raw.values
    # X = X[:-1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=238194)

    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    
    # clf.fit(X_train, y_train)
    
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)
    
    # print(y_pred)
    # print(y_test)
    # time = np.linspace(1,10000,1000)
    # plt.plot(time, y_raw)
    # plt.show()
    
    now = datetime.now()
    time_delta = timedelta(seconds=100)
    print(now+time_delta)
    
    

    