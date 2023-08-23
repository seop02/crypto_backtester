from sklearn.ensemble import RandomForestClassifier

import ax
from ax.service.managed_loop import optimize
import numpy as np
import train_model
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

def  optimize_loop(parameters, evaluation_function):
    best_parameters, best_values, experiemt, model = optimize(
        parameters=parameters,
        evaluation_function=evaluation_function,
        objective_name='accuracy',
        total_trials=25  # Number of optimization iterations
        )
    #storing the optimal results and hyperparameters
    means, covariances = best_values
    return best_parameters, means

def  optimize_parameters(parameters, classifiers, X_train, X_test, y_train, y_test):
    accuracy = np.zeros((len(classifiers)))    
    prediction = np.zeros((len(classifiers), len(y_test)))

    best_hyperparameters = {}

    for i in range(len(classifiers)):
            if classifiers[i] == "LogisticR_lbfgs":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_liblinear":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "LogisticR_newtonc":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_newtonch":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "LogisticR_sag":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_saga":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
    

            
            elif classifiers[i] == "LDA_le":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LDA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.LDA(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "QDA":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.QDA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.QDA(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "BernoulliNB":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.BNB(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.BNB(X_train, X_test, y_train, y_test, 'final', best_parameters)
                
            elif classifiers[i] == "SVM":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.SVM(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.SVM(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "AdaBoost":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.ADA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.ADA(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "RandomForest":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.RF(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.RF(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "KNN":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.KNN(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.KNN(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "DecisionTree":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.DT(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                accuracy[i] = means['accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, prediction[i] = train_model.DT(X_train, X_test, y_train, y_test, 'final', best_parameters)

            else: 
                print("error!")

            d = {'classifiers': classifiers, 'BA': accuracy}
            df = pd.DataFrame(data=d)
    return df