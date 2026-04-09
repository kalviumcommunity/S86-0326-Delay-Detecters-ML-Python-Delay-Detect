"""Unit tests for baseline model utilities."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import (
    compare_to_baseline,
    evaluate_classification_baseline,
    evaluate_regression_baseline,
    train_classification_baseline,
    train_regression_baseline,
)


class TestBaselineClassification(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
        })
        # Imbalanced target: majority class = 0
        self.y_train = np.array([0] * 80 + [1] * 20)

        self.X_test = pd.DataFrame({
            "f1": np.random.randn(30),
            "f2": np.random.randn(30),
        })
        self.y_test = np.array([0] * 24 + [1] * 6)

    def test_majority_baseline_predicts_single_majority_class(self):
        baseline = train_classification_baseline(
            self.X_train,
            self.y_train,
            strategy="most_frequent",
        )
        pred = baseline.predict(self.X_test)

        self.assertEqual(len(np.unique(pred)), 1)
        self.assertEqual(np.unique(pred)[0], 0)

    def test_classification_baseline_metrics_have_expected_keys(self):
        baseline = train_classification_baseline(self.X_train, self.y_train)
        metrics = evaluate_classification_baseline(baseline, self.X_test, self.y_test)

        expected = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        self.assertEqual(set(metrics.keys()), expected)

    def test_stratified_baseline_outputs_valid_labels(self):
        baseline = train_classification_baseline(
            self.X_train,
            self.y_train,
            strategy="stratified",
        )
        pred = baseline.predict(self.X_test)

        self.assertEqual(len(pred), len(self.X_test))
        self.assertTrue(set(np.unique(pred)).issubset({0, 1}))


class TestBaselineRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 3)
        self.y_train = np.random.normal(50, 10, size=100)

        self.X_test = np.random.randn(25, 3)
        self.y_test = np.random.normal(50, 10, size=25)

    def test_mean_baseline_predicts_constant_close_to_training_mean(self):
        baseline = train_regression_baseline(self.X_train, self.y_train, strategy="mean")
        pred = baseline.predict(self.X_test)

        self.assertEqual(len(np.unique(np.round(pred, 8))), 1)
        self.assertAlmostEqual(pred[0], float(np.mean(self.y_train)), places=6)

    def test_regression_baseline_metrics_have_expected_keys(self):
        baseline = train_regression_baseline(self.X_train, self.y_train, strategy="median")
        metrics = evaluate_regression_baseline(baseline, self.X_test, self.y_test)

        expected = {"mse", "rmse", "mae", "r2"}
        self.assertEqual(set(metrics.keys()), expected)


class TestBaselineComparison(unittest.TestCase):
    def test_compare_to_baseline_returns_metric_deltas(self):
        baseline_metrics = {"accuracy": 0.70, "f1": 0.65, "roc_auc": None}
        model_metrics = {"accuracy": 0.81, "f1": 0.74, "roc_auc": None}

        improvement = compare_to_baseline(model_metrics, baseline_metrics)

        self.assertAlmostEqual(improvement["accuracy"], 0.11, places=7)
        self.assertAlmostEqual(improvement["f1"], 0.09, places=7)
        self.assertIsNone(improvement["roc_auc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
