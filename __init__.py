import os

api_path = f"{os.path.dirname(__file__)}"


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