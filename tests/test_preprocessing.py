"""
Unit Tests for data_preprocessing module.

These tests demonstrate how modular structure enables isolated testing.
Each function can be tested independently without running the entire pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the functions we want to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    load_data,
    handle_missing_values,
    remove_duplicates,
    split_data
)


class TestDataPreprocessing(unittest.TestCase):
    """
    Test suite for data preprocessing functions.
    
    BENEFITS OF MODULAR STRUCTURE:
    - Can test load_data() without running train()
    - Can test handle_missing_values() without loading actual data
    - Can test split_data() without preprocessing
    - Easy to catch bugs in isolation
    - Easy to refactor with confidence
    """
    
    def setUp(self):
        """Set up test data before each test."""
        # Create larger test dataframe (30+ rows) for stratified split to work
        np.random.seed(42)
        n_samples = 30
        self.df_simple = pd.DataFrame({
            'distance_km': np.random.uniform(10, 100, n_samples),
            'items_count': np.random.randint(1, 10, n_samples),
            'zone': np.random.choice(['A', 'B', 'C'], n_samples),
            'is_delayed': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # Imbalanced like real data
        })
        
        # Create second dataframe with nulls for testing
        self.df_with_nulls = pd.DataFrame({
            'distance_km': [10, np.nan, 30, 40, np.nan, 50, 60, 70, np.nan, 80] * 3,
            'items_count': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10] * 3,
            'zone': ['A', 'B', 'A', np.nan, 'B', 'C', 'A', np.nan, 'B', 'C'] * 3,
            'is_delayed': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3
        })
        
        self.df_with_duplicates = pd.DataFrame({
            'distance_km': [10, 20, 20, 30, 30, 30],
            'items_count': [1, 2, 2, 3, 3, 3],
            'is_delayed': [0, 1, 1, 0, 0, 0]
        })
    
    def test_handle_missing_values_median(self):
        """Test that handle_missing_values fills NaNs with median strategy."""
        df_clean = handle_missing_values(self.df_with_nulls, strategy='median')
        
        # Assert no null values remain in numerical columns
        self.assertEqual(df_clean[['distance_km', 'items_count']].isnull().sum().sum(), 0)
        
        # Assert categorical column was filled (with mode)
        self.assertEqual(df_clean['zone'].isnull().sum(), 0)
    
    def test_handle_missing_values_no_nulls_after(self):
        """Test that function removes all nulls."""
        df_clean = handle_missing_values(self.df_with_nulls, strategy='median')
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
    
    def test_remove_duplicates(self):
        """Test that remove_duplicates removes duplicate rows."""
        df_clean = remove_duplicates(self.df_with_duplicates)
        
        # Should remove duplicates
        self.assertEqual(len(df_clean), 3)  # Only 3 unique rows
        self.assertTrue(df_clean.duplicated().sum() == 0)
    
    def test_split_data_creates_four_outputs(self):
        """Test that split_data returns four elements."""
        X_train, X_test, y_train, y_test = split_data(
            self.df_simple,
            target_column='is_delayed',
            test_size=0.2
        )
        
        # Should have all four components
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
    
    def test_split_data_respects_test_size(self):
        """Test that split_data respects test_size parameter."""
        X_train, X_test, y_train, y_test = split_data(
            self.df_simple,
            target_column='is_delayed',
            test_size=0.2,
            random_state=42
        )
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        # Test ratio should be approximately 0.2
        self.assertAlmostEqual(test_ratio, 0.2, places=1)
    
    def test_split_data_target_excluded_from_x(self):
        """Test that target column is not in X."""
        X_train, X_test, y_train, y_test = split_data(
            self.df_simple,
            target_column='is_delayed'
        )
        
        # Target column should not be in X
        self.assertNotIn('is_delayed', X_train.columns)
        self.assertNotIn('is_delayed', X_test.columns)
        
        # But should be in y
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))


class TestDataPreprocessingIntegration(unittest.TestCase):
    """
    Integration tests showing how to use multiple preprocessing functions together.
    These demonstrate the pipeline flow.
    """
    
    def test_preprocessing_pipeline_flow(self):
        """Test typical preprocessing workflow."""
        # Create sample data with issues - use 30+ rows for stratified split
        np.random.seed(42)
        n_samples = 30
        df = pd.DataFrame({
            'distance_km': list(np.random.uniform(10, 100, n_samples-1)) + [np.nan],  # One NaN for testing
            'items_count': list(range(1, n_samples+1)),
            'is_delayed': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        })
        
        # Step 1: Remove duplicates
        initial_len = len(df)
        df = remove_duplicates(df)
        self.assertLessEqual(len(df), initial_len)  # Could remove some duplicates
        
        # Step 2: Handle missing values
        df = handle_missing_values(df, strategy='median')
        self.assertEqual(df.isnull().sum().sum(), 0)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column='is_delayed',
            test_size=0.2
        )
        
        # Assertions
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train) + len(X_test), len(df))


# ============================================================================
# HOW TO RUN THESE TESTS
# ============================================================================

"""
Command line:
    cd d:/ML-python
    python -m pytest tests/test_preprocessing.py -v

Or using unittest:
    cd d:/ML-python
    python -m unittest tests.test_preprocessing -v

Or directly:
    cd d:/ML-python
    python tests/test_preprocessing.py

BENEFITS:
✅ Test individual functions in isolation
✅ Catch bugs before they reach production
✅ Refactor with confidence - tests catch regressions
✅ Document expected behavior
✅ Enable continuous integration
"""

if __name__ == '__main__':
    unittest.main(verbosity=2)
