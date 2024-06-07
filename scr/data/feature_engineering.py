
# import pandas as pd

# def create_features(df):
#     df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
#     df['TransactionHour'] = df['TransactionStartTime'].dt.hour
#     df['TransactionDay'] = df['TransactionStartTime'].dt.day
#     df['TransactionMonth'] = df['TransactionStartTime'].dt.month
#     df['TransactionYear'] = df['TransactionStartTime'].dt.year
#     df = df.drop(columns=['TransactionStartTime'])
#     return df
import pandas as pd

def create_aggregate_features(data):
    data['TotalTransactionAmount'] = data.groupby('CustomerId')['Amount'].transform('sum')
    data['AverageTransactionAmount'] = data.groupby('CustomerId')['Amount'].transform('mean')
    data['TransactionCount'] = data.groupby('CustomerId')['TransactionId'].transform('count')
    data['StdTransactionAmount'] = data.groupby('CustomerId')['Amount'].transform('std')
    return data

def extract_date_features(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    return data

if __name__ == "__main__":
    data = pd.read_csv('data/raw/data.csv')
    data = create_aggregate_features(data)
    data = extract_date_features(data)
    print(data.head())
