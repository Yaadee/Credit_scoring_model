import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib  # You can also use `import pickle`

def train_model(X, y, model_path='model.pkl'):
    # Ensure X is a DataFrame before dropping columns
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X.toarray())  # Convert sparse matrix to DataFrame if necessary

    target = 'FraudResult'  # Replace with the correct target column if different
    if target in X.columns:
        X = X.drop(columns=[target])
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the trained model to a pickle file
    joblib.dump(model, model_path)  # Use pickle.dump(model, open(model_path, 'wb')) if using pickle
    
    return model
