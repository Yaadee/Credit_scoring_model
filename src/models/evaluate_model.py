
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    evaluation_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred)
    }
    return evaluation_metrics

if __name__ == "__main__":
    from train_model import train_model
    import pandas as pd
    
    data = pd.read_csv('creditScoring/data/processed/processed.csv')
    X = data.drop('FraudResult', axis=1)
    y = data['FraudResult']
    
    model, X_test, y_test = train_model(X, y)
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

