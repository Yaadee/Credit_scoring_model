import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xverse.transformer import WOE

def create_aggregate_features(df):
    df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['AverageTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['TransactionCount'] = df.groupby('CustomerId')['Amount'].transform('count')
    df['StdDevTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')
    return df

def extract_datetime_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['TransactionHour'] = df[date_column].dt.hour
    df['TransactionDay'] = df[date_column].dt.day
    df['TransactionMonth'] = df[date_column].dt.month
    df['TransactionYear'] = df[date_column].dt.year
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def handle_missing_values(df):
    df = df.fillna(df.median())
    return df

def normalize_numerical_features(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def calculate_rfm(df, reference_date):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (reference_date - x.max()).days,  # Recency
        'CustomerId': 'count',                                             # Frequency
        'Amount': ['sum', 'mean']                                          # Monetary and Spend
    }).reset_index()
    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary', 'spend']
    rfm.fillna(0, inplace=True)
    return rfm

def calculate_rfms_score(rfm):
    rfm['rfms_score'] = (0.25 * rfm['recency']) + (0.25 * rfm['frequency']) + (0.25 * rfm['monetary']) + (0.25 * rfm['spend'])
    return rfm

def classify_users(rfm, threshold):
    rfm['user_class'] = rfm['rfms_score'].apply(lambda x: 'good' if x >= threshold else 'bad')
    return rfm

def perform_woe_binning(rfm):
    woe_transformer = WOE()
    woe_transformer.fit(rfm.drop(columns=['CustomerId', 'user_class']), rfm['user_class'])
    rfm_woe = woe_transformer.transform(rfm.drop(columns=['CustomerId', 'user_class']))
    return rfm_woe

def calculate_iv(feature, target):
    # Placeholder function for IV calculation
    iv = 0.1  # Placeholder value
    return iv

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
    useful_features = [feature for feature, iv in iv_values.items() if iv >= 0.02]
    
    data = data[useful_features]  # Include only useful features
    
    # Perform WoE binning
    data_woe = perform_woe_binning(data)
    
    return data_woe

if __name__ == "__main__":
    data = pd.read_csv('creditScoring/data/raw/data.csv')
    data_preprocessed = preprocess_data(data)
    print(data_preprocessed.shape)


