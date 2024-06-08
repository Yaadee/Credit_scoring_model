import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Load the processed DataFrame
df_processed = pd.read_csv('creditScoring/data/processed/data_preprocessed.csv')

# Overview of the Data
def data_overview(df):
    print("Overview of the Data:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nData Types:\n", df.dtypes)

# Summary Statistics
def summary_statistics(df):
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

# Distribution of Numerical Features
def numerical_feature_distribution(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    num_features_count = len(numerical_features)
    rows = math.ceil(num_features_count / 3)
    plt.figure(figsize=(15, rows * 5))
    for i, feature in enumerate(numerical_features):
        plt.subplot(rows, 3, i + 1)
        sns.histplot(df[feature], bins=15, kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

# Distribution of Categorical Features
def categorical_feature_distribution(df):
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)
        plt.title(f'Distribution of {feature}')
        plt.show()

# Correlation Analysis
def correlation_analysis(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Identifying Missing Values
def missing_values(df):
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing.sort_values(ascending=False))

# Outlier Detection
def outlier_detection(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.show()
def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()


# Perform EDA
data_overview(df_processed)
summary_statistics(df_processed)
numerical_feature_distribution(df_processed)
categorical_feature_distribution(df_processed)
correlation_analysis(df_processed)
missing_values(df_processed)
outlier_detection(df_processed)
plot_distribution


