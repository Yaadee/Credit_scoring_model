import pandas as pd
from feature_engineering import create_aggregate_features, extract_datetime_features, encode_categorical_variables, handle_missing_values, normalize_numerical_features, calculate_rfm, calculate_rfms_score, classify_users, perform_woe_binning

def calculate_iv(feature, target):
    # Placeholder function for IV calculation
    iv = 0.1  # Placeholder value
    
    if iv < 0.02:
        return 'not useful'
    elif 0.02 <= iv < 0.1:
        return 'weak relationship'
    elif 0.1 <= iv < 0.3:
        return 'medium strength relationship'
    elif 0.3 <= iv < 0.5:
        return 'strong relationship'
    else:
        return 'suspicious relationship'

def preprocess_data(data):
    data = create_aggregate_features(data)
    data = extract_datetime_features(data, 'TransactionStartTime')
    data = encode_categorical_variables(data)
    data = handle_missing_values(data)
    data = normalize_numerical_features(data)
    
    # Calculate IV for categorical variables
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    iv_values = {}
    for feature in categorical_features:
        iv_values[feature] = calculate_iv(data[feature], data['FraudResult'])
    
    # Filter features based on IV thresholds
    useful_features = [feature for feature, iv_label in iv_values.items() if iv_label in ['medium strength relationship', 'strong relationship', 'suspicious relationship']]
    
    data = data[useful_features]  # Include only useful features
    
    # Perform WoE binning
    data_woe = perform_woe_binning(data)
    
    return data_woe

if __name__ == "__main__":
    data = pd.read_csv('creditScoring/data/raw/data.csv')
    data_preprocessed = preprocess_data(data)
    print(data_preprocessed.shape)







# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import pandas as pd

# def preprocess_data(data):
#     categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
#     numerical_features = ['Amount', 'Value']

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='mean')),
#                 ('scaler', StandardScaler())]), numerical_features),
#             ('cat', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
#         ])

#     data_preprocessed = preprocessor.fit_transform(data)
    
#     # Convert the sparse matrix to a dense array
#     return data_preprocessed.toarray()

# if __name__ == "__main__":
#     data = pd.read_csv('data/raw/data.csv')
#     data_preprocessed = preprocess_data(data)
#     print(data_preprocessed.shape)


# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import pandas as pd
# from xverse.transformer import WOE

# def preprocess_data(data, target_column):
#     categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
#     numerical_features = ['Amount', 'Value']
    
#     # WoE transformation for categorical features
#     woe = WOE()
#     woe.fit(data[categorical_features], data[target_column])
#     data_woe = woe.transform(data[categorical_features])

#     # Combine WoE-transformed categorical features with numerical features
#     data_combined = pd.concat([data_woe, data[numerical_features]], axis=1)
    
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='mean')),
#                 ('scaler', StandardScaler())]), numerical_features),
#             ('cat', 'passthrough', data_woe.columns)
#         ])
    
#     data_preprocessed = preprocessor.fit_transform(data_combined)
    
#     # Convert the sparse matrix to a dense array
#     return data_preprocessed

# if __name__ == "__main__":
#     data = pd.read_csv('creditScoring/data/raw/data.csv')
#     data_preprocessed = preprocess_data(data, target_column='FraudResult')
#     print(data_preprocessed.shape)

