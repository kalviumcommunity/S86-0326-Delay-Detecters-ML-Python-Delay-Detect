"""
Unit Tests for feature_engineering module.

Demonstrates how modular structure with clear responsibilities
enables focused, meaningful unit tests.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    encode_categorical_features,
    scale_numerical_features,
    build_preprocessing_pipeline
)


class TestFeatureEngineering(unittest.TestCase):
    """
    Test suite for feature engineering functions.
    
    WHY THIS MATTERS:
    - Each function can be tested independently
    - Can verify encoding works correctly
    - Can verify scaling preserves relationships
    - Can test pipeline building without fitting
    - Easy to catch feature engineering bugs
    """
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'distance_km': [10.0, 20.0, 30.0, 40.0, 50.0],
            'items_count': [1, 2, 3, 4, 5],
            'zone': ['A', 'B', 'A', 'C', 'B'],
            'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        })
        
        self.categorical_cols = ['zone', 'day_of_week']
        self.numerical_cols = ['distance_km', 'items_count']
    
    def test_encode_categorical_features_output_shape(self):
        """Test that encoding produces correct output shape."""
        df_encoded = encode_categorical_features(
            self.df,
            categorical_cols=self.categorical_cols,
            method='onehot'
        )
        
        # Encoding should produce more columns (one-hot expansion)
        self.assertGreater(len(df_encoded.columns), len(self.df.columns))
        
        # Should have same number of rows
        self.assertEqual(len(df_encoded), len(self.df))
    
    def test_encode_removes_original_categorical_columns(self):
        """Test that original categorical columns are replaced."""
        df_encoded = encode_categorical_features(
            self.df,
            categorical_cols=self.categorical_cols,
            method='onehot'
        )
        
        # Original categorical columns should not be in encoded data
        for col in self.categorical_cols:
            self.assertNotIn(col, df_encoded.columns)
    
    def test_scale_numerical_features(self):
        """Test that scaling transforms features correctly."""
        df_scaled, scaler = scale_numerical_features(
            self.df,
            numerical_cols=self.numerical_cols,
            fit=True
        )
        
        # Scaled numerical columns should have mean ~0 and std ~1
        for col in self.numerical_cols:
            # Check that values are in reasonable range after scaling
            self.assertLess(abs(df_scaled[col].mean()), 1.0)
    
    def test_scale_fit_vs_transform(self):
        """Test that fit=True and fit=False work correctly."""
        # Fit on first DataFrame
        df_subset = self.df.head(3)
        df_scaled_fit, scaler = scale_numerical_features(
            df_subset,
            numerical_cols=self.numerical_cols,
            fit=True
        )
        
        # Transform on full DataFrame using fitted scaler
        df_scaled_transform, _ = scale_numerical_features(
            self.df,
            numerical_cols=self.numerical_cols,
            scaler=scaler,
            fit=False
        )
        
        # Both should have numerical columns
        self.assertIn(self.numerical_cols[0], df_scaled_fit.columns)
        self.assertIn(self.numerical_cols[0], df_scaled_transform.columns)

    def test_scale_numerical_features_minmax_range(self):
        """Test MinMax scaling keeps training numerical values in [0, 1]."""
        df_scaled, _ = scale_numerical_features(
            self.df,
            numerical_cols=self.numerical_cols,
            scaler_type='minmax',
            fit=True
        )

        for col in self.numerical_cols:
            self.assertGreaterEqual(df_scaled[col].min(), 0.0)
            self.assertLessEqual(df_scaled[col].max(), 1.0)
    
    def test_build_preprocessing_pipeline_returns_pipeline(self):
        """Test that building pipeline returns a pipeline object."""
        pipeline = build_preprocessing_pipeline(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        
        # Should have transformers attribute
        self.assertTrue(hasattr(pipeline, 'transformers'))
        
        # Should have fit_transform method
        self.assertTrue(hasattr(pipeline, 'fit_transform'))
    
    def test_pipeline_fit_transform_output_shape(self):
        """Test fit_transform produces numerical output."""
        pipeline = build_preprocessing_pipeline(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(self.df)
        
        # Output should be numpy array
        self.assertTrue(isinstance(X_transformed, np.ndarray))
        
        # Should have same number of rows
        self.assertEqual(X_transformed.shape[0], len(self.df))
    
    def test_pipeline_transform_only(self):
        """Test that pipeline can be fitted once and reused."""
        # Create two DataFrames
        df_train = self.df.head(3)
        df_test = self.df.tail(2)
        
        # Build and fit pipeline on training data
        pipeline = build_preprocessing_pipeline(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        X_train = pipeline.fit_transform(df_train)
        
        # Apply to test data without refitting
        X_test = pipeline.transform(df_test)
        
        # Both should have same number of features (same columns)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        # But different number of samples
        self.assertEqual(X_train.shape[0], len(df_train))
        self.assertEqual(X_test.shape[0], len(df_test))


class TestFeatureEngineeringWorkflow(unittest.TestCase):
    """
    Demonstrates proper usage: Fit on training data, apply to test data.
    This prevents data leakage.
    """
    
    def test_proper_train_test_workflow(self):
        """
        Demonstrate the CORRECT way:
        - Fit pipeline on training data
        - Apply pipeline to test data (transform only)
        - This prevents data leakage
        """
        # Sample data
        df = pd.DataFrame({
            'distance_km': [10, 20, 30, 40, 50],
            'zone': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Split manually for demonstration
        df_train = df.head(3)
        df_test = df.tail(2)
        
        # CORRECT: Fit pipeline on TRAINING data
        pipeline = build_preprocessing_pipeline(
            categorical_cols=['zone'],
            numerical_cols=['distance_km']
        )
        
        X_train = pipeline.fit_transform(df_train)
        
        # CORRECT: Apply to TEST data (transform only, no fitting)
        X_test = pipeline.transform(df_test)
        
        # Verify shapes
        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        print("✅ Proper workflow: Pipeline fit on training, applied to test")
        print(f"   Training shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")


# ============================================================================
# TESTING BEST PRACTICES DEMONSTRATED
# ============================================================================

"""
WHY MODULAR STRUCTURE ENABLES TESTING:

1. UNIT TESTING (test individual functions):
   - test_encode_categorical_features() tests encoding in isolation
   - test_scale_numerical_features() tests scaling in isolation  
   - test_build_preprocessing_pipeline() tests pipeline building in isolation
   
   Without modular structure, you CAN'T test these independently!

2. INTEGRATION TESTING (test combined workflow):
   - test_pipeline_fit_transform_output_shape() tests complete transform
   - test_proper_train_test_workflow() tests train/test separation
   
   Verifies that components work together correctly.

3. REGRESSION TESTING (ensure changes don't break things):
   - Run tests before and after refactoring
   - Catch bugs introduced by changes immediately
   - Refactor with confidence

4. DOCUMENTATION:
   - Tests document expected behavior
   - Show how to use functions correctly
   - Prevent misuse

5. MAINTAINABILITY:
   - Easy to locate bugs (which test fails?)
   - Easy to fix bugs (isolated to specific function)
   - Easy to add features (new tests document new behavior)
"""

if __name__ == '__main__':
    unittest.main(verbosity=2)
