# import pandas as pd

# def load_data(data_path, variable_definitions_path):
#     data = pd.read_csv(data_path)
#     variable_definitions = pd.read_csv(variable_definitions_path)
#     return data, variable_definitions
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    data = load_data('data/raw/data.csv')
    print(data.head())
