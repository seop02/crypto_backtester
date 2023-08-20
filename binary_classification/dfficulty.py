from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
import json
import logging
import os
from preprocessing import preprocessing

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


def hvdm_std(std):
    def hvdm(x, y):
        result = np.abs(x-y)/(4*std)
        return np.sum(np.sqrt(result))
    return hvdm
#'hepatitis', 'iris', 'echocardiogram', 'wine', 'parkinson', 'appendictis', 'yeast', 'glass'

if __name__ == '__main__':
    difficulty = {}



    path = 'data_types.json'
    if not os.path.exists(path):
        json_object = json.dumps(difficulty, indent=4)
    
    # Writing to sample.json
        with open(path, "w") as outfile:
            outfile.write(json_object)



    scale = 10
    duration = 8640
    X, y = preprocessing(scale, duration)
    if X[y==0].shape[0]> X[y==1].shape[0]:
        c = 1
    else:
        c = 0

    d = {"safe": 0, "borderline": 0, "rare":0, "outlier":0}
    rkf = RepeatedKFold(n_splits=10, n_repeats=5)
    test = []
    std = np.std(X, axis=0)

    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        LOG.debug(f'Current fold is {i}')
        knn = KNeighborsClassifier(n_neighbors=5, metric=hvdm_std(std))
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        k_indices = knn.kneighbors(X_test, return_distance=False)

        for j in range(len(y_test)):
            identical = 0
            
            if y_test[j] == c:
                for k in range(5):
                    if y_train[k_indices[j][k]] == y_test[j]:
                        identical+=1
                if identical>=4:
                    d["safe"] += 1
                elif 4>identical>=2:
                    d["borderline"] += 1
                elif 2>identical >= 1:
                    d["rare"] += 1
                else:
                    d["outlier"] += 1

    normalize = sum(d.values())

    difficulty = {key: value / normalize for key, value in d.items()}


    with open(path, "w") as json_file:
        json.dump(difficulty, json_file, indent=4)