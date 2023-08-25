import numpy as np
import pandas as pd
from preprocessing import preprocessing
import logging
from sklearn.model_selection import train_test_split
from optimize_loop import optimize_parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
import sys
sys.path.append("./")

from src.BaseSVDD import BaseSVDD


logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    # parameters = {
    # "LogisticR_lbfgs": {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ['none', 'l2'],
    # 'solver': ['lbfgs']
    # },
    # "LogisticR_liblinear":  {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ['l1', 'l2'],
    # 'solver': ['liblinear']
    # },
    # "LogisticR_newtonc": {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ['none', 'l2'],
    # 'solver': ['newton-cg']
    # },
    # "LogisticR_newtonch": {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ['none', 'l2'],
    # 'solver': ['newton-cholesky']
    # },
    # "LogisticR_sag": {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ['none', 'l2'],
    # 'solver': ['sag']
    # },
    # "LogisticR_saga": {
    # 'C': [0.01, 0.1, 1, 10, 100],
    # 'penalty': ["l2", "elasticnet", "l1", "None"],
    # 'solver': ['saga']
    # },
    # "LDA_le": [
    # {"name": "solver", "type": "choice", "values": ["lsqr", "eigen"]},
    # {"name": "shrinkage", "type": "range", "bounds": [0.0, 1.0]}
    # ],
    # "QDA": [
    # {"name": "reg_param", "type": "range", "bounds": [0.0, 1.0]}
    # ],
    # "BernoulliNB": [
    # {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
    # {"name": "binarize", "type": "range", "bounds": [0.0, 1.0]}
    # ],
    # "AdaBoost": [
    # {"name": "n_estimators", "type": "range", "bounds": [50, 200]},
    # {"name": "learning_rate", "type": "range", "bounds": [0.1, 1.0]}
    # ],
    # "RandomForest": {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # },
    # "KNN": [
    # {"name": "weights", "type": "choice", "values": ["uniform", "distance"]},
    # {"name": "n_neighbours", "type": "range", "bounds": [1, 10]}
    # ],
    # "DecisionTree": {
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    # }

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

    cutoff_space = [0.0006]#np.linspace(0.0001, 0.001, 10)
    scale_space = [30]
    for scale in scale_space:
        for cutoff in cutoff_space:
            duration = 8640
            LOG.info(f"Current sclae is {scale}, current cutoff is {cutoff}")
            X, y = preprocessing(scale, duration, cutoff)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=None)

            LOG.info(len(y_train))
            LOG.info(len(y_test))


            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            LOG.info(f"RandomForest Test Accuracy:{accuracy}")
            
            classifiers = [
                "DecisionTree", "RandomForest", "KNN"]
            df = optimize_parameters(
                parameters, classifiers, X_train, X_test, y_train, y_test
                )

            df.to_csv(f"optimized_{scale}_{duration}_final_{cutoff:.4f}.csv")
            



            # LOG.info("DecisionTree Best Hyperparameters:", dt_grid_search.best_params_)
            # LOG.info("DecisionTree Test Accuracy:", accuracy_dt)


