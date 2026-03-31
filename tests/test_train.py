"""
Unit Tests for training module.

Demonstrates how proper encapsulation and modular design
enables effective unit testing of model training logic.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_model
from sklearn.ensemble import RandomForestClassifier


class TestTraining(unittest.TestCase):
    """
    Test suite for model training functions.
    
    KEY INSIGHT:
    Because train_model() accepts data as parameters and returns
    a model object, it can be tested in isolation without:
    - Loading actual data
    - Reading from disk
    - Running preprocessing
    - Any other side effects
    
    This is only possible with proper encapsulation!
    """
    
    def setUp(self):
        """Create synthetic training data for testing."""
        np.random.seed(42)
        
        # Simple synthetic data
        self.X_train = np.random.randn(100, 5)  # 100 samples, 5 features
        self.y_train = np.random.randint(0, 2, 100)  # Binary classification
        
        # Slightly different test data (different samples)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_train_model_returns_fitted_model(self):
        """Test that train_model returns a fitted model object."""
        model = train_model(self.X_train, self.y_train)
        
        # Should return a model object
        self.assertIsNotNone(model)
        
        # Should be the correct type
        self.assertIsInstance(model, RandomForestClassifier)
        
        # Should be fitted (have n_classes_ attribute)
        self.assertTrue(hasattr(model, 'n_classes_'))
    
    def test_train_model_produces_predictions(self):
        """Test that trained model can make predictions."""
        model = train_model(self.X_train, self.y_train)
        
        # Should be able to predict
        predictions = model.predict(self.X_test)
        
        # Should have same number of predictions as samples
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Predictions should be binary (0 or 1)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
    
    def test_train_model_respects_random_state(self):
        """Test that random_state produces reproducible results."""
        # Train two models with same random state
        model1 = train_model(self.X_train, self.y_train, random_state=42)
        model2 = train_model(self.X_train, self.y_train, random_state=42)
        
        # Get predictions from both
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)
        
        # Predictions should be identical (reproducibility)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_train_model_different_random_states_may_differ(self):
        """Test that different random states may produce different results."""
        model1 = train_model(self.X_train, self.y_train, random_state=42)
        model2 = train_model(self.X_train, self.y_train, random_state=123)
        
        # Get predictions
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)
        
        # Models were trained with different seeds,
        # so predictions may differ (though might coincidentally match)
        # We just verify both produce valid predictions
        self.assertEqual(len(pred1), len(self.X_test))
        self.assertEqual(len(pred2), len(self.X_test))
    
    def test_train_model_with_different_hyperparams(self):
        """Test that train_model accepts hyperparameters."""
        # Train with custom hyperparameters
        model = train_model(
            self.X_train,
            self.y_train,
            model_type="random_forest",
            random_state=42,
            n_estimators=50,
            max_depth=5
        )
        
        # Should have the hyperparameters we specified
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 5)
    
    def test_train_model_data_validation(self):
        """Test that train_model validates input data."""
        # X and y have different lengths
        X_bad = np.random.randn(100, 5)
        y_bad = np.random.randint(0, 2, 50)  # Wrong length!
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            train_model(X_bad, y_bad)


class TestTrainingBestPractices(unittest.TestCase):
    """
    Demonstrates testing best practices enabled by good design.
    """
    
    def setUp(self):
        """Synthetic data."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_model_training_workflow(self):
        """
        Test complete workflow:
        1. Train model
        2. Evaluate model
        3. Verify model quality
        
        This demonstrates how separate functions enable testing workflows.
        """
        # Step 1: Train model
        model = train_model(self.X_train, self.y_train, random_state=42)
        self.assertIsNotNone(model)
        
        # Step 2: Evaluate model
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        
        # Step 3: Verify quality
        # Model should do better than random guessing (0.5 for binary)
        self.assertGreater(train_score, 0.5)
        
        # Should have reasonable test performance (with some margin for randomness)
        self.assertGreaterEqual(test_score, 0.25)
        
        print(f"✅ Training workflow test passed")
        print(f"   Train score: {train_score:.3f}")
        print(f"   Test score: {test_score:.3f}")
    
    def test_model_does_not_improve_from_noise(self):
        """
        Test that model learns real patterns, not noise.
        
        If we give model random data with random labels,
        it should perform near random chance.
        """
        # Random features, random labels
        X_random = np.random.randn(100, 5)
        y_random = np.random.randint(0, 2, 100)
        
        model = train_model(X_random, y_random, random_state=42)
        
        # For random data, test performance should be around 50% (or lower)
        test_score = model.score(np.random.randn(20, 5), np.random.randint(0, 2, 20))
        
        # Shouldn't be much better than random
        self.assertLess(test_score, 0.7)  # Performance on random data
        
        print(f"✅ Random data test passed")
        print(f"   Model performance on random data: {test_score:.3f} (expected ~0.5)")


# ============================================================================
# TESTING PRINCIPLES DEMONSTRATED
# ============================================================================

"""
WHY UNIT TESTS ARE POSSIBLE WITH GOOD DESIGN:

GOOD DESIGN (Testable):
    def train_model(X_train, y_train, random_state=42):
        # Inputs explicit as parameters
        # Output explicit as return value
        # No side effects (doesn't read/write files)
        # No hidden dependencies
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        return model  # Return the artifact

✅ TESTABLE because:
  - Can call with synthetic data
  - Can verify return value type and content
  - Can test reproducibility with random_state
  - Can test parameter passing
  - No side effects to worry about

BAD DESIGN (Not testable):
    def train_model():
        # No parameters! Where does data come from?
        df = pd.read_csv('data.csv')        # Implicit file dependency
        X = df.drop('target', axis=1)
        y = df['target']
        model = RandomForest(random_state=GLOBAL_SEED)  # Global dependency
        model.fit(X, y)
        save_model(model)                   # Side effect: writes file!
        plot_results(model)                 # Side effect: creates plot!
        # No return value!

❌ NOT TESTABLE because:
  - Can't call with test data (no parameters)
  - Depends on specific file existing
  - Depends on global variable
  - Has side effects (writes files, creates plots)
  - No return value to verify

CONCLUSION:
Good structure isn't just for code cleanliness.
It's ESSENTIAL for testing, debugging, and maintainability!
"""

if __name__ == '__main__':
    unittest.main(verbosity=2)
