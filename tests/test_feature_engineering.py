import unittest
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.data.feature_engineering import (
    create_aggregate_features, 
    extract_datetime_features, 
    encode_categorical_variables, 
    handle_missing_values, 
    normalize_numerical_features, 
    calculate_rfm, 
    calculate_rfms_score, 
    classify_users, 
    perform_woe_binning
)

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        df, _ = load_data('data/data.csv', 'data/Xente_Variable_Definitions.csv')
        df = preprocess_data(df)
        self.df = create_aggregate_features(df)
        self.df = extract_datetime_features(self.df, 'TransactionDate')
        self.df = encode_categorical_variables(self.df)
        self.df = handle_missing_values(self.df)
        self.df = normalize_numerical_features(self.df)

    def test_calculate_rfm(self):
        reference_date = pd.Timestamp('2023-12-31')
        rfm = calculate_rfm(self.df, reference_date)
        self.assertIn('recency', rfm.columns)
        self.assertIn('frequency', rfm.columns)
        self.assertIn('monetary', rfm.columns)
        self.assertIn('spend', rfm.columns)

    def test_calculate_rfms_score(self):
        reference_date = pd.Timestamp('2023-12-31')
        rfm = calculate_rfm(self.df, reference_date)
        rfm = calculate_rfms_score(rfm)
        self.assertIn('rfms_score', rfm.columns)

    def test_classify_users(self):
        reference_date = pd.Timestamp('2023-12-31')
        rfm = calculate_rfm(self.df, reference_date)
        rfm = calculate_rfms_score(rfm)
        rfm = classify_users(rfm, threshold=0.5)
        self.assertIn('user_class', rfm.columns)

    def test_perform_woe_binning(self):
        reference_date = pd.Timestamp('2023-12-31')
        rfm = calculate_rfm(self.df, reference_date)
        rfm = calculate_rfms_score(rfm)
        rfm = classify_users(rfm, threshold=0.5)
        rfm_woe = perform_woe_binning(rfm)
        self.assertIn('user_class', rfm_woe.columns)

if __name__ == '__main__':
    unittest.main()
