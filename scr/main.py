# # src/main.py
# from data.load_data import load_data
# from data.preprocess_data import preprocess_data
# from data.feature_engineering import create_features
# from models.train_model import train_model
# from models.evaluate_model import evaluate_model
# import pandas as pd

# if __name__ == "__main__":
#     data, variable_definitions = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/raw/Xente_Variable_Definitions.csv')
    
#     data = create_features(data)
#     data_processed = preprocess_data(data)
    
#     # Splitting the data into features and target
#     X = data_processed
#     y = data['FraudResult']  # Adjust according to your target column name
    
#     # Train the model
#     model = train_model(X, y)
    
#     # Evaluate the model
#     metrics = evaluate_model(model, X, y)
#     print(metrics)
from data.load_data import load_data
from data.preprocess_data import preprocess_data
from data.feature_engineering import create_features
from models.train_model import train_model
from models.evaluate_model import evaluate_model
import pandas as pd

if __name__ == "__main__":
    data, variable_definitions = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/raw/Xente_Variable_Definitions.csv')
    
    data = create_features(data)
    data_processed = preprocess_data(data)
    
    # Ensure data_processed is a DataFrame
    if not isinstance(data_processed, pd.DataFrame):
        data_processed = pd.DataFrame(data_processed)

    # Splitting the data into features and target
    X = data_processed
    y = data['FraudResult']  # Adjust according to your target column name
    
    # Train the model and save it to 'model.pkl'
    model = train_model(X, y, model_path='creditScoring/scr/models/model.pkl')
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    print(metrics)

