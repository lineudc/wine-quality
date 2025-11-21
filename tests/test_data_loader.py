import unittest
import pandas as pd
from src.data_loader import load_and_combine_data

class TestDataLoader(unittest.TestCase):
    def test_load_and_combine_data(self):
        df = load_and_combine_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('type', df.columns)
        self.assertTrue(len(df) > 0)
        self.assertTrue(all(df['type'].isin(['red', 'white'])))

if __name__ == '__main__':
    unittest.main()
