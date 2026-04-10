"""Unit tests for MSE/R2 regression evaluation utilities."""

import unittest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mse_r2_evaluation import (
    compare_model_vs_baseline_mse_r2,
    cross_validate_mse_rmse,
    cross_validate_r2,
    evaluate_regression_with_mse_r2,
)
from src.regression import build_linear_regression_pipeline


class TestMseR2Evaluation(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.y_pred = np.array([12.0, 18.0, 33.0, 37.0, 48.0])

        X = rng.randn(150, 4)
        noise = rng.normal(0, 0.4, size=150)
        y = 1.8 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2] + noise

        self.X_train = X
        self.y_train = y

    def test_evaluate_regression_with_mse_r2_keys(self):
        metrics = evaluate_regression_with_mse_r2(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"mse", "rmse", "r2"})

    def test_compare_model_vs_baseline_mse_r2_outputs(self):
        comparison = compare_model_vs_baseline_mse_r2(
            model_metrics={"mse": 25.0, "rmse": 5.0, "r2": 0.7},
            baseline_metrics={"mse": 49.0, "rmse": 7.0, "r2": 0.0},
        )

        self.assertAlmostEqual(comparison["mse_reduction"], 24.0, places=7)
        self.assertAlmostEqual(comparison["rmse_reduction"], 2.0, places=7)
        self.assertAlmostEqual(comparison["r2_gain"], 0.7, places=7)

    def test_cross_validate_r2_returns_expected_shape(self):
        model = build_linear_regression_pipeline(scale_features=True)
        result = cross_validate_r2(model, self.X_train, self.y_train, cv=5)

        self.assertEqual(len(result["scores"]), 5)
        self.assertGreaterEqual(result["mean"], -1.0)

    def test_cross_validate_mse_rmse_returns_positive_scores(self):
        model = build_linear_regression_pipeline(scale_features=True)
        result = cross_validate_mse_rmse(model, self.X_train, self.y_train, cv=5)

        self.assertEqual(len(result["mse_scores"]), 5)
        self.assertEqual(len(result["rmse_scores"]), 5)
        self.assertGreaterEqual(result["mse_mean"], 0)
        self.assertGreaterEqual(result["rmse_mean"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
