
# import unittest
# from load_data import load_data
# from preprocess_data import preprocess_data
# from feature_engineering import create_features

# class TestDataLoading(unittest.TestCase):
#     def test_load_data(self):
#         data, variable_definitions = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/raw/Xente_Variable_Definitions.csv')
#         self.assertIsNotNone(data)
#         self.assertIsNotNone(variable_definitions)
#         self.assertFalse(data.empty)
#         self.assertFalse(variable_definitions.empty)

#     def test_preprocess_data(self):
#         data, _ = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/raw/Xente_Variable_Definitions.csv')
#         data = create_features(data)
#         data_processed = preprocess_data(data)
#         self.assertIsNotNone(data_processed)

# if __name__ == '__main__':
#     unittest.main()
# tests/test_data.py

import unittest
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data

class TestData(unittest.TestCase):
    def test_load_data(self):
        df, definitions = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/Xente_Variable_Definitions.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(definitions, pd.DataFrame)

    def test_preprocess_data(self):
        df, _ = load_data('creditScoring/data/raw/data.csv', 'creditScoring/data/raw/Xente_Variable_Definitions.csv')
        df = preprocess_data(df)
        self.assertFalse(df.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
