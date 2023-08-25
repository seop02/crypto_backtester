from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np

def RF(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_RF(hyperparameters):
            # Create and train your RandomForestClassifier
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=hyperparameters['n_estimators'],
                max_depth=hyperparameters['max_depth'],
                min_samples_split=hyperparameters['min_samples_split']
            )
            model.fit(X_train, y_train)
            
            # Evaluate the model's performance
            predictions = model.predict(X_test)
            score = balanced_accuracy_score(y_test, predictions)
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

            return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_RF
    
    elif mode == 'final':
        return train_RF(hyperparameters)

def LogisticR(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def logisticR(hyperparameters):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=hyperparameters['C'],
            solver=hyperparameters['solver'],
            max_iter=hyperparameters['max_iter'])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Evaluate the model's performance
        
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return logisticR
        
    elif mode == 'final':
        return logisticR(hyperparameters)


def LDA(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_LDA(hyperparameters):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(solver=hyperparameters['solver'],
            shrinkage=hyperparameters['shrinkage'])
        model.fit(X_train, y_train)
        
        # Evaluate the model's performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Evaluate the model's performance
        
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_LDA
        
    elif mode == 'final':
        return train_LDA(hyperparameters)
    
def QDA(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_QDA(hyperparameters):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis(reg_param=hyperparameters['reg_param'])
        model.fit(X_train, y_train)

        # Evaluate the model's performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_QDA
        
    elif mode == 'final':
        return train_QDA(hyperparameters)

def BNB(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_BernoulliNB(hyperparameters):
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB(alpha=hyperparameters['alpha'], 
                            binarize=hyperparameters['binarize'])
        model.fit(X_train, y_train)

        # Evaluate the model's performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_BernoulliNB
        
    elif mode == 'final':
        return train_BernoulliNB(hyperparameters)

def SVM(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_SVM(hyperparameters):
        from sklearn.svm import SVC
        model = SVC(kernel=hyperparameters['kernel'], 
                    C=hyperparameters['C'],
                    gamma=hyperparameters['gamma'])

        # Evaluate the model's performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_SVM
        
    elif mode == 'final':
        return train_SVM(hyperparameters)

def ADA(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_ada(hyperparameters):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=hyperparameters['n_estimators'], 
                    learning_rate=hyperparameters['learning_rate'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return score, tn, fp, fn, tp

    if mode == 'optimize':
        return train_ada
        
    elif mode == 'final':
        return train_ada(hyperparameters)

def KNN(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_KNN(hyperparameters):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(weights=hyperparameters['weights'], 
                    n_neighbors=hyperparameters['n_neighbours'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_KNN
        
    elif mode == 'final':
        return train_KNN(hyperparameters)

def DT(X_train, X_test, y_train, y_test, mode, hyperparameters):
    def train_DT(hyperparameters):
        from sklearn import tree
        model = tree.DecisionTreeClassifier(criterion=hyperparameters['criterion'], 
                    max_depth=hyperparameters['max_depth'],
                    min_samples_split=hyperparameters['min_samples_split'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return score, tn, fp, fn, tp
    if mode == 'optimize':
        return train_DT
        
    elif mode == 'final':
        return train_DT(hyperparameters)

