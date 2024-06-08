import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from feature_engineering import preprocess_data

def train_model(X, y, model, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    
    return metrics

if __name__ == "__main__":
    data = pd.read_csv('creditScoring/data/processed/data_preprocessed.csv')
    X = data.drop('FraudResult', axis=1)
    y = data['FraudResult']
    
    # Preprocess data
    X = preprocess_data(X)
    
    # Logistic Regression
    lr_param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
    lr_model = LogisticRegression()
    lr_best_model, X_test, y_test = train_model(X, y, lr_model, lr_param_grid)
    lr_metrics = evaluate_model(lr_best_model, X_test, y_test)
    print("Logistic Regression:", lr_metrics)
    
    # Decision Tree
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    dt_model = DecisionTreeClassifier()
    dt_best_model, X_test, y_test = train_model(X, y, dt_model, dt_param_grid)
    dt_metrics = evaluate_model(dt_best_model, X_test, y_test)
    print("Decision Tree:", dt_metrics)
    
    # Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestClassifier()
    rf_best_model, X_test, y_test = train_model(X, y, rf_model, rf_param_grid)
    rf_metrics = evaluate_model(rf_best_model, X_test, y_test)
    print("Random Forest:", rf_metrics)
    
    # Gradient Boosting Machines
    gbm_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gbm_model = GradientBoostingClassifier()
    gbm_best_model, X_test, y_test = train_model(X, y, gbm_model, gbm_param_grid)
    gbm_metrics = evaluate_model(gbm_best_model, X_test, y_test)
    print("Gradient Boosting Machines:", gbm_metrics)














# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd

# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
    
#     grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
#     grid_search.fit(X_train, y_train)
    
#     best_rf = grid_search.best_estimator_
#     return best_rf, X_test, y_test

# if __name__ == "__main__":
#     data = pd.read_csv('creditScoring/data/processed/data_preprocessed.csv')
#     X = data.drop('FraudResult', axis=1)
#     y = data['FraudResult']
#     model, X_test, y_test = train_model(X, y)
#     print("Model trained!")
