from data.load_data import load_data
from data.preprocess_data import preprocess_data
from data.feature_engineering import create_aggregate_features, extract_date_features
from models.train_model import train_model
from models.evaluate_model import evaluate_model
import pandas as pd

def main():
    # Load data
    data = load_data('data/raw/data.csv')
    
    # Feature engineering
    data = create_aggregate_features(data)
    data = extract_date_features(data)
    
    # Preprocess data
    data_preprocessed = preprocess_data(data)
    
    # Convert preprocessed data back to DataFrame for saving
    data_preprocessed_df = pd.DataFrame(data_preprocessed)
    data_preprocessed_df['FraudResult'] = data['FraudResult'].values
    
    # Save processed data
    data_preprocessed_df.to_csv('data/processed/data_preprocessed.csv', index=False)
    
    # Load processed data
    processed_data = pd.read_csv('data/processed/data_preprocessed.csv')
    X = processed_data.drop('FraudResult', axis=1)
    y = processed_data['FraudResult']
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)
    
if __name__ == "__main__":
    main()

