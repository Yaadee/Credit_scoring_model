
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
#     data = pd.read_csv('data/processed/data_preprocessed.csv')
#     X = data.drop('FraudResult', axis=1)
#     y = data['FraudResult']
#     model, X_test, y_test = train_model(X, y)
#     print("Model trained!")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    return best_rf, X_test, y_test

if __name__ == "__main__":
    data = pd.read_csv('data/processed/data_preprocessed.csv')
    X = data.drop('FraudResult', axis=1)
    y = data['FraudResult']
    model, X_test, y_test = train_model(X, y)
    print("Model trained!")
