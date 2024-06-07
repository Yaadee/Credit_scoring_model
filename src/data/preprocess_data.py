# # import pandas as pd
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer

# # # Import the custom load_data function from load_data.py
# # from load_data import load_data

# # # Load your dataset using the custom function
# # data_path = 'data/raw/data.csv'  # Replace with the path to your data file
# # variable_definitions_path = 'data/raw/Xente_Variable_Definitions.csv'  # Replace with the path to your variable definitions file
# # df, variable_definitions = load_data(data_path, variable_definitions_path)

# # def extract_numeric_parts(df):
# #     df['TransactionId'] = df['TransactionId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['BatchId'] = df['BatchId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['AccountId'] = df['AccountId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['SubscriptionId'] = df['SubscriptionId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['CustomerId'] = df['CustomerId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['ProviderId'] = df['ProviderId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['ProductId'] = df['ProductId'].str.extract('(\d+)', expand=False).astype(float)
# #     df['ChannelId'] = df['ChannelId'].str.extract('(\d+)', expand=False).astype(float)

# # def preprocess_df(df):
# #     # Separate numeric and categorical columns
# #     numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
# #     categorical_features = df.select_dtypes(include=['object']).columns

# #     # Define preprocessing for numeric features: impute missing values and scale
# #     numeric_transformer = Pipeline(steps=[
# #         ('imputer', SimpleImputer(strategy='median')),
# #         ('scaler', StandardScaler())
# #     ])

# #     # Define preprocessing for categorical features: impute missing values and one-hot encode with max categories
# #     categorical_transformer = Pipeline(steps=[
# #         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
# #         ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False))  # Limit categories and ensure dense output
# #     ])

# #     # Combine preprocessing steps
# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ('num', numeric_transformer, numeric_features),
# #             ('cat', categorical_transformer, categorical_features)
# #         ])

# #     # Apply the transformations to the dataset
# #     df_processed = preprocessor.fit_transform(df)
    
# #     # Convert the processed dataset back to a DataFrame
# #     df_processed = pd.DataFrame(df_processed, columns=numeric_features.tolist() + 
# #                                                     preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist())
    
# #     return df_processed

# # # Apply the functions
# # extract_numeric_parts(df)
# # df_processed = preprocess_df(df)

# # # Save the processed DataFrame to a CSV file
# # df_processed.to_csv('data/processed/processed.csv', index=False)

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
#     return data_preprocessed

# if __name__ == "__main__":
#     data = pd.read_csv('data/raw/data.csv')
#     data_preprocessed = preprocess_data(data)
#     print(data_preprocessed.shape)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def preprocess_data(data):
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    numerical_features = ['Amount', 'Value']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

    data_preprocessed = preprocessor.fit_transform(data)
    
    # Convert the sparse matrix to a dense array
    return data_preprocessed.toarray()

if __name__ == "__main__":
    data = pd.read_csv('data/raw/data.csv')
    data_preprocessed = preprocess_data(data)
    print(data_preprocessed.shape)
