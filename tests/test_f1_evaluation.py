"""Unit tests for F1-focused classification evaluation helpers."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.f1_evaluation import (
    compare_f1_vs_baseline,
    cross_validate_f1,
    evaluate_binary_f1_metrics,
    evaluate_f1_averages,
    evaluate_fbeta,
    find_best_threshold_for_f1,
    threshold_predictions,
)
from src.logistic_classification import build_logistic_pipeline


class TestF1Evaluation(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        self.y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.6])

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
            0.05 * X_num["distance_km"]
            + 0.04 * X_num["items_count"]
            + 0.003 * X_num["order_value"]
            + (X_cat["peak_hour"] == "Evening").astype(int) * 0.4
            + rng.normal(0, 0.9, size=n)
        )
        self.y = (signal > np.median(signal)).astype(int)

        self.num_cols = ["distance_km", "items_count", "order_value", "day_of_month"]
        self.cat_cols = ["zone", "day_of_week", "peak_hour"]

    def test_evaluate_binary_f1_metrics_keys(self):
        metrics = evaluate_binary_f1_metrics(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"precision", "recall", "f1"})

    def test_evaluate_f1_averages_keys(self):
        metrics = evaluate_f1_averages(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"f1_micro", "f1_macro", "f1_weighted"})

    def test_threshold_predictions_binary(self):
        pred = threshold_predictions(self.y_prob, threshold=0.5)
        self.assertTrue(set(np.unique(pred)).issubset({0, 1}))

    def test_find_best_threshold_for_f1_contract(self):
        best = find_best_threshold_for_f1(self.y_true, self.y_prob)
        self.assertIn("best_threshold", best)
        self.assertIn("best_f1", best)

    def test_compare_f1_vs_baseline(self):
        gain = compare_f1_vs_baseline(model_f1=0.4, baseline_f1=0.2)
        self.assertAlmostEqual(gain["f1_gain"], 0.2, places=7)

    def test_cross_validate_f1_shapes(self):
        model = build_logistic_pipeline(
            categorical_cols=self.cat_cols,
            numerical_cols=self.num_cols,
            max_iter=1000,
            random_state=42,
        )
        result = cross_validate_f1(model, self.X, self.y, cv=5, random_state=42)

        self.assertEqual(len(result["f1_binary_scores"]), 5)
        self.assertEqual(len(result["f1_macro_scores"]), 5)
        self.assertEqual(len(result["f1_weighted_scores"]), 5)

    def test_fbeta_non_negative(self):
        score = evaluate_fbeta(self.y_true, self.y_pred, beta=2.0)
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
