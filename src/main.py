import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.data.feature_engineering import (
    create_aggregate_features, 
    extract_datetime_features, 
    encode_categorical_variables, 
    handle_missing_values, 
    normalize_numerical_features, 
    calculate_rfm, 
    calculate_rfms_score, 
    classify_users, 
    perform_woe_binning
)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rfms_space(rfm):
    sns.scatterplot(data=rfm, x='recency', y='frequency', hue='user_class')
    plt.title('RFMS Space')
    plt.show()

def main_pipeline(file_path, reference_date, threshold):
    df = load_data(file_path)
    df = create_aggregate_features(df)
    df = extract_datetime_features(df, 'TransactionDate')
    df = encode_categorical_variables(df)
    df = handle_missing_values(df)
    df = normalize_numerical_features(df)
    rfm = calculate_rfm(df, reference_date)
    rfm = calculate_rfms_score(rfm)
    rfm = classify_users(rfm, threshold)
    plot_rfms_space(rfm)
    rfm_woe = perform_woe_binning(rfm)
    return rfm_woe

if __name__ == "__main__":
    file_path = 'data/data.csv'
    reference_date = pd.Timestamp('2023-12-31')
    threshold = 0.5
    rfm_woe = main_pipeline(file_path, reference_date, threshold)
    print(rfm_woe.head())












# # this is good compared to the first on 

# from data.load_data import load_data
# from data.preprocess_data import preprocess_data
# from data.feature_engineering import create_aggregate_features, extract_date_features
# from models.train_model import train_model
# from models.evaluate_model import evaluate_model
# import pandas as pd

# def main():
#     # Load data
#     data = load_data('creditScoring/data/raw/data.csv')
    
#     # Feature engineering
#     data = create_aggregate_features(data)
#     data = extract_date_features(data)
    
#     # Preprocess data
#     data_preprocessed = preprocess_data(data)
    
#     # Convert preprocessed data back to DataFrame for saving
#     data_preprocessed_df = pd.DataFrame(data_preprocessed)
#     data_preprocessed_df['FraudResult'] = data['FraudResult'].values
    
#     # Save processed data
#     data_preprocessed_df.to_csv('data/processed/data_preprocessed.csv', index=False)
    
#     # Load processed data
#     processed_data = pd.read_csv('data/processed/data_preprocessed.csv')
#     X = processed_data.drop('FraudResult', axis=1)
#     y = processed_data['FraudResult']
    
#     # Train model
#     model, X_test, y_test = train_model(X, y)
    
#     # Evaluate model
#     metrics = evaluate_model(model, X_test, y_test)
#     print(metrics)
    
# if __name__ == "__main__":
#     main()




# from data.load_data import load_data
# from data.preprocess_data import preprocess_data
# from data.feature_engineering import create_aggregate_features, extract_date_features, encode_woe
# from models.train_model import train_model
# from models.evaluate_model import evaluate_model
# import pandas as pd

# def main():
#     # Load data
#     data = load_data('data/raw/data.csv')
    
#     # Feature engineering
#     data = create_aggregate_features(data)
#     data = extract_date_features(data)
    
#     # Preprocess data
#     data_preprocessed = preprocess_data(data, target_column='FraudResult')
    
#     # Convert preprocessed data back to DataFrame for saving
#     data_preprocessed_df = pd.DataFrame(data_preprocessed)
#     data_preprocessed_df['FraudResult'] = data['FraudResult'].values
    
#     # Save processed data
#     data_preprocessed_df.to_csv('data/processed/data_preprocessed.csv', index=False)
    
#     # Load processed data
#     processed_data = pd.read_csv('data/processed/data_preprocessed.csv')
#     X = processed_data.drop('FraudResult', axis=1)
#     y = processed_data['FraudResult']
    
#     # Train model
#     model, X_test, y_test = train_model(X, y)
    
#     # Evaluate model
#     metrics = evaluate_model(model, X_test, y_test)
#     print(metrics)
    
# if __name__ == "__main__":
#     main()

