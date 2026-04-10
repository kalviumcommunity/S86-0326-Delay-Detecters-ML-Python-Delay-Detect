"""Unit tests for MAE-focused regression evaluation utilities."""

import unittest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mae_evaluation import (
    compare_model_vs_baseline_mae,
    cross_validate_mae,
    evaluate_regression_with_mae,
    mae_as_percentage_of_mean_target,
    residual_diagnostics,
)
from src.regression import build_linear_regression_pipeline


class TestMaeEvaluation(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.y_pred = np.array([12.0, 18.0, 33.0, 37.0, 48.0])

        X = rng.randn(120, 4)
        noise = rng.normal(0, 0.2, size=120)
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + noise
        self.X_train = X
        self.y_train = y

    def test_evaluate_regression_with_mae_keys(self):
        metrics = evaluate_regression_with_mae(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"mae", "mse", "rmse", "r2"})

    def test_mae_percentage_positive(self):
        metrics = evaluate_regression_with_mae(self.y_true, self.y_pred)
        pct = mae_as_percentage_of_mean_target(metrics["mae"], self.y_true)
        self.assertGreater(pct, 0)

    def test_compare_model_vs_baseline_mae_sign(self):
        comp = compare_model_vs_baseline_mae(model_mae=7.5, baseline_mae=10.0)
        self.assertAlmostEqual(comp["mae_improvement"], 2.5, places=7)
        self.assertAlmostEqual(comp["mae_improvement_pct"], 25.0, places=7)

    def test_cross_validate_mae_returns_positive_scores(self):
        model = build_linear_regression_pipeline(scale_features=True)
        result = cross_validate_mae(model, self.X_train, self.y_train, cv=5)

        self.assertEqual(len(result["scores"]), 5)
        self.assertGreaterEqual(result["mean"], 0)
        self.assertGreaterEqual(result["std"], 0)

    def test_residual_diagnostics_contains_bias_stats(self):
        diag = residual_diagnostics(self.y_true, self.y_pred)
        self.assertIn("residual_mean", diag)
        self.assertIn("residual_median", diag)
        self.assertIn("residual_std", diag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
