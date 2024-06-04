import pandas as pd

def load_data(data_path, variable_definitions_path):
    data = pd.read_csv(data_path)
    variable_definitions = pd.read_csv(variable_definitions_path)
    return data, variable_definitions
