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
        objective_name='balanced_accuracy',
        total_trials=15  # Number of optimization iterations
        )
    #storing the optimal results and hyperparameters
    means, covariances = best_values
    return best_parameters, means

def  optimize_parameters(parameters, classifiers, X_train, X_test, y_train, y_test):
    balanced_accuracy = np.zeros((len(classifiers)))    
    tn_list = np.zeros((len(classifiers)))
    fp_list = np.zeros((len(classifiers)))
    fn_list = np.zeros((len(classifiers)))
    tp_list = np.zeros((len(classifiers)))

    best_hyperparameters = {}

    for i in range(len(classifiers)):
            if classifiers[i] == "LogisticR_lbfgs":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_liblinear":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "LogisticR_newtonc":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_newtonch":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "LogisticR_sag":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "LogisticR_saga":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LogisticR(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LogisticR(X_train, X_test, y_train, y_test, 'final', best_parameters)
    

            
            elif classifiers[i] == "LDA_le":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.LDA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.LDA(X_train, X_test, y_train, y_test, 'final', best_parameters)
            
            elif classifiers[i] == "QDA":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.QDA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.QDA(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "BernoulliNB":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.BNB(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.BNB(X_train, X_test, y_train, y_test, 'final', best_parameters)
                
            elif classifiers[i] == "SVM":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.SVM(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.SVM(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "AdaBoost":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.ADA(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.ADA(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "RandomForest":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.RF(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.RF(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "KNN":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.KNN(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.KNN(X_train, X_test, y_train, y_test, 'final', best_parameters)

            elif classifiers[i] == "DecisionTree":
                LOG.info(f"current classifier is {classifiers[i]}")
                best_parameters, means = optimize_loop(parameters=parameters[classifiers[i]], 
                                                       evaluation_function=
                                                       train_model.DT(X_train, X_test, y_train, y_test, 'optimize', parameters[classifiers[i]]))
                balanced_accuracy[i] = means['balanced_accuracy']
                best_hyperparameters[f'{classifiers[i]}'] = best_parameters
                score, tn_list[i], fp_list[i], fn_list[i], tp_list[i] = train_model.DT(X_train, X_test, y_train, y_test, 'final', best_parameters)

            else: 
                print("error!")

            d = {'clssifiers': classifiers, 'BA': balanced_accuracy, 'TN': tn_list, 'FP': fp_list, 'FN': fn_list, 'TP': tp_list}
            df = pd.DataFrame(data=d)
    return df