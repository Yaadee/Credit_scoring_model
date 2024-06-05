# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_histogram(df, column):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df[column], kde=True, bins=30)
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

# def plot_boxplot(df, column):
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df[column])
#     plt.title(f'Boxplot of {column}')
#     plt.xlabel(column)
#     plt.show()

# def plot_correlation_matrix(df):
#     plt.figure(figsize=(12, 8))
#     correlation_matrix = df.corr()
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title('Correlation Matrix')
#     plt.show()

# def plot_categorical_distribution(df, column):
#     plt.figure(figsize=(10, 6))
#     sns.countplot(y=column, data=df, order=df[column].value_counts().index)
#     plt.title(f'Distribution of {column}')
#     plt.xlabel('Frequency')
#     plt.ylabel(column)
#     plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def plot_categorical_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(y=column, data=df, order=df[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Frequency')
    plt.ylabel(column)
    plt.show()

