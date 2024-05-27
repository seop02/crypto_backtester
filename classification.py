import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from backtest import data_path

if __name__=='__main__':
    X = np.load(f'{data_path}/binary/x_tot.npy')
    y = np.load(f'{data_path}/binary/y_tot.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(n_estimators = 100,random_state=42)

    # Perform cross-validation
    scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='balanced_accuracy')
    print("Cross-Validation Accuracy Scores:", scores)
    print("Mean Cross-Validation Accuracy:", scores.mean())
    
    xgb_clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Test Set Accuracy:", accuracy)
    print("Test Set F1-Score:", f1)
    #print(X_test)
