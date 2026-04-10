"""Unit tests for logistic classification utilities."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_classification import (
    build_logistic_pipeline,
    compare_model_vs_baseline,
    cross_validate_auc_f1,
    evaluate_binary_classifier,
    extract_logistic_coefficients,
    train_majority_baseline,
)


class TestLogisticClassification(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        n = 240

        X_num = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 50, size=n),
                "items_count": rng.randint(1, 20, size=n),
                "order_value": rng.uniform(10, 500, size=n),
                "day_of_month": rng.randint(1, 31, size=n),
            }
        )
        X_cat = pd.DataFrame(
            {
                "zone": rng.choice(["Zone_A", "Zone_B", "Zone_C"], size=n),
                "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n),
                "peak_hour": rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n),
            }
        )

        self.X = pd.concat([X_num, X_cat], axis=1)

        signal = (
            0.05 * self.X["distance_km"]
            + 0.08 * self.X["items_count"]
            + 0.005 * self.X["order_value"]
            + (self.X["peak_hour"] == "Evening").astype(int) * 0.4
            + rng.normal(0, 0.8, size=n)
        )
        self.y = (signal > np.median(signal)).astype(int)

        self.num_cols = ["distance_km", "items_count", "order_value", "day_of_month"]
        self.cat_cols = ["zone", "day_of_week", "peak_hour"]

    def test_majority_baseline_predicts_constant_class(self):
        baseline = train_majority_baseline(self.X, self.y)
        pred = baseline.predict(self.X.head(30))
        self.assertEqual(len(np.unique(pred)), 1)

    def test_pipeline_fit_and_metrics(self):
        pipe = build_logistic_pipeline(
            categorical_cols=self.cat_cols,
            numerical_cols=self.num_cols,
            max_iter=1000,
            random_state=42,
        )
        pipe.fit(self.X, self.y)

        pred = pipe.predict(self.X)
        prob = pipe.predict_proba(self.X)[:, 1]
        metrics = evaluate_binary_classifier(self.y, pred, prob)

        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_compare_model_vs_baseline_has_expected_keys(self):
        gains = compare_model_vs_baseline(
            model_metrics={"accuracy": 0.80, "f1": 0.78, "roc_auc": 0.86},
            baseline_metrics={"accuracy": 0.62, "f1": 0.50, "roc_auc": 0.50},
        )

        self.assertEqual(set(gains.keys()), {"accuracy_gain", "f1_gain", "roc_auc_gain"})
        self.assertGreater(gains["roc_auc_gain"], 0)

    def test_cross_validate_auc_f1_shapes(self):
        pipe = build_logistic_pipeline(
            categorical_cols=self.cat_cols,
            numerical_cols=self.num_cols,
            max_iter=1000,
            random_state=42,
        )
        result = cross_validate_auc_f1(pipe, self.X, self.y, cv=5)

        self.assertEqual(len(result["auc_scores"]), 5)
        self.assertEqual(len(result["f1_scores"]), 5)

    def test_extract_coefficients_outputs_odds_ratios(self):
        pipe = build_logistic_pipeline(
            categorical_cols=self.cat_cols,
            numerical_cols=self.num_cols,
            max_iter=1000,
            random_state=42,
        )
        pipe.fit(self.X, self.y)

        coef_df = extract_logistic_coefficients(pipe)
        self.assertIn("feature", coef_df.columns)
        self.assertIn("coefficient", coef_df.columns)
        self.assertIn("odds_ratio", coef_df.columns)
        self.assertTrue((coef_df["odds_ratio"] > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
