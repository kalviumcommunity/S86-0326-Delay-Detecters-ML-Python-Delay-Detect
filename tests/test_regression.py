"""Unit tests for linear regression utilities."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.regression import (
    build_linear_regression_pipeline,
    compare_model_to_baseline,
    cross_validate_r2,
    evaluate_regression_model,
    get_linear_coefficients,
    train_regression_baseline,
)


class TestRegressionUtilities(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 50, size=180),
                "items_count": rng.randint(1, 20, size=180),
                "order_value": rng.uniform(10, 500, size=180),
                "day_of_month": rng.randint(1, 31, size=180),
            }
        )

        noise = rng.normal(0, 2.0, size=180)
        self.y = (
            12.0
            + 1.35 * self.X["distance_km"]
            + 0.95 * self.X["items_count"]
            + 0.015 * self.X["order_value"]
            + 0.25 * self.X["day_of_month"]
            + noise
        )

    def test_baseline_can_fit_and_predict(self):
        baseline = train_regression_baseline(self.X, self.y, strategy="mean")
        pred = baseline.predict(self.X.head(10))

        self.assertEqual(len(pred), 10)
        self.assertEqual(len(np.unique(np.round(pred, 8))), 1)

    def test_linear_regression_pipeline_improves_over_baseline(self):
        baseline = train_regression_baseline(self.X, self.y, strategy="mean")
        baseline_pred = baseline.predict(self.X)
        baseline_metrics = evaluate_regression_model(self.y, baseline_pred)

        pipeline = build_linear_regression_pipeline(scale_features=True)
        pipeline.fit(self.X, self.y)
        model_pred = pipeline.predict(self.X)
        model_metrics = evaluate_regression_model(self.y, model_pred)

        self.assertLess(model_metrics["rmse"], baseline_metrics["rmse"])
        self.assertGreater(model_metrics["r2"], baseline_metrics["r2"])

    def test_cross_validation_returns_expected_structure(self):
        pipeline = build_linear_regression_pipeline(scale_features=True)
        cv_result = cross_validate_r2(pipeline, self.X, self.y, cv=5)

        self.assertIn("scores", cv_result)
        self.assertIn("mean", cv_result)
        self.assertIn("std", cv_result)
        self.assertEqual(len(cv_result["scores"]), 5)

    def test_coefficient_extraction_matches_feature_count(self):
        pipeline = build_linear_regression_pipeline(scale_features=True)
        pipeline.fit(self.X, self.y)

        coef_df = get_linear_coefficients(pipeline, list(self.X.columns))
        self.assertEqual(len(coef_df), len(self.X.columns))
        self.assertIn("feature", coef_df.columns)
        self.assertIn("coefficient", coef_df.columns)

    def test_compare_model_to_baseline_has_expected_keys(self):
        improvements = compare_model_to_baseline(
            model_metrics={"mse": 4.0, "rmse": 2.0, "mae": 1.5, "r2": 0.7},
            baseline_metrics={"mse": 9.0, "rmse": 3.0, "mae": 2.0, "r2": 0.0},
        )

        expected_keys = {
            "mse_improvement",
            "rmse_improvement",
            "mae_improvement",
            "r2_improvement",
        }
        self.assertEqual(set(improvements.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
