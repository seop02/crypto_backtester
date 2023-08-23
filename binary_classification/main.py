import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from preprocessing import preprocessing
from optimize_loop import optimize_parameters
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


if __name__ == '__main__':
    
    parameters = {
    "LogisticR_lbfgs": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["lbfgs"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l2", "None"]}
    ],
    "LogisticR_liblinear": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["liblinear"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l1", "l2"]}
    ],
    "LogisticR_newtonc": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["newton-cg"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l2", "None"]}
    ],
    "LogisticR_newtonch": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["newton-cholesky"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l2", "None"]}
    ],
    "LogisticR_sag": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["sag"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l2", "None"]}
    ],
    "LogisticR_saga": [
    {"name": "C", "type": "range", "bounds": [0.1, 10.0]},
    {"name": "solver", "type": "choice", "values": ["saga"]},
    {"name": "max_iter", "type": "range", "bounds": [100, 1000]},
    {"name": "penalty", "type": "choice", "values": ["l2", "elasticnet", "l1", "None"]}
    ],
    "LDA_le": [
    {"name": "solver", "type": "choice", "values": ["lsqr", "eigen"]},
    {"name": "shrinkage", "type": "range", "bounds": [0.0, 1.0]}
    ],
    "QDA": [
    {"name": "reg_param", "type": "range", "bounds": [0.0, 1.0]}
    ],
    "BernoulliNB": [
    {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
    {"name": "binarize", "type": "range", "bounds": [0.0, 1.0]}
    ],
    "AdaBoost": [
    {"name": "n_estimators", "type": "range", "bounds": [50, 200]},
    {"name": "learning_rate", "type": "range", "bounds": [0.1, 1.0]}
    ],
    "RandomForest": [
    {"name": "n_estimators", "type": "range", "bounds": [50, 200]},
    {"name": "max_depth", "type": "range", "bounds": [5, 20]},
    {"name": "min_samples_split", "type": "range", "bounds": [2, 10]},
    ],
    "KNN": [
    {"name": "weights", "type": "choice", "values": ["uniform", "distance"]},
    {"name": "n_neighbours", "type": "range", "bounds": [1, 10]}
    ],
    "DecisionTree": [
    {"name": "criterion", "type": "choice", "values": ["gini", "entropy", "log_loss"]},
    {"name": "max_depth", "type": "range", "bounds": [1, 10]},
    {"name": "min_samples_split", "type": "range", "bounds": [2, 10]}
    ]}
    cutoff_space = [0.0003]#np.linspace(0.0001, 0.001, 10)
    scale_space = [60]
    for scale in scale_space:
        for cutoff in cutoff_space:
            duration = 8640
            LOG.info(f"Current sclae is {scale}, current cutoff is {cutoff}")
            X, y = preprocessing(scale, duration, cutoff)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            data = pd.DataFrame({'actual': y_test})
            data.to_csv('actual.csv')

            LOG.info(len(y_train))
            LOG.info(len(y_test))
            # clf = RandomForestClassifier()
            # clf.fit(X_train, y_train)
            # y_pred = clf.predict(X_test)
            # accuracy = accuracy_score(y_pred, y_test)
            # print(accuracy)
            # data = pd.DataFrame(y_pred)
            # data.to_csv('trial_data.csv')

            # classifiers = [
            #     "DecisionTree", "RandomForest", "KNN"
            #             ]
            # df = optimize_parameters(
            #     parameters, classifiers, X_train, X_test, y_train, y_test
            #     )

            # df.to_csv(f"optimized_{scale}_{duration}_final_{cutoff:.4f}.csv")
    

    