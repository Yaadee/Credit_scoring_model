import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Separate numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Define preprocessing for numeric features: impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical features: impute missing values and one-hot encode with max categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False))  # Limit categories and ensure dense output
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply the transformations to the dataset
    df_processed = preprocessor.fit_transform(df)
    
    # Convert the processed data back to a DataFrame
    df_processed = pd.DataFrame(df_processed, columns=numeric_features.tolist() + 
                                                    preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist())
    
    return df_processed
