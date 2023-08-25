import numpy as np
import pandas as pd
from preprocessing import preprocessing, one_class
from optimize_loop import optimize_parameters
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.model_selection import learning_curve, GridSearchCV
import sys
sys.path.append("./")

from src.BaseSVDD import BaseSVDD

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)



if __name__ == '__main__':

    #one class verification
    cutoff = 0.0006
    scale = 30
    #0.0003, 10
    #0.0005, 30
    
    duration = 8640
    LOG.info(f"Current sclae is {scale}, current cutoff is {cutoff}")
    X, y = one_class(scale, duration, cutoff)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    nu = y_test.count(-1)/len(y_test)
    LOG.info(nu)
    model = OneClassSVM(nu=nu)


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = balanced_accuracy_score(y_test, y_pred)
    LOG.info(f"best score: {accuracy}")
    LOG.info(np.count_nonzero(y_pred==1))
    LOG.info(np.count_nonzero(y_pred==-1))
    LOG.info(y_test.count(1))
    LOG.info(y_test.count(-1))

    #binary classification

