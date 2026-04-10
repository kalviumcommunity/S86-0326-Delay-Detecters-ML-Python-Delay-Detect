"""Unit tests for precision/recall evaluation helpers."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_classification import build_logistic_pipeline
from src.precision_recall_evaluation import (
    best_threshold_for_min_recall,
    cross_validate_precision_recall,
    evaluate_fbeta,
    evaluate_precision_recall,
    precision_recall_curve_data,
    threshold_predictions,
)


class TestPrecisionRecallEvaluation(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        self.y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.6])

        rng = np.random.RandomState(42)
        n = 220
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

    def test_evaluate_precision_recall_keys(self):
        metrics = evaluate_precision_recall(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"precision", "recall", "f1"})

    def test_threshold_predictions_binary_output(self):
        pred = threshold_predictions(self.y_prob, threshold=0.5)
        self.assertTrue(set(np.unique(pred)).issubset({0, 1}))

    def test_precision_recall_curve_data_shapes(self):
        curve = precision_recall_curve_data(self.y_true, self.y_prob)
        self.assertIn("precisions", curve)
        self.assertIn("recalls", curve)
        self.assertIn("thresholds", curve)
        self.assertEqual(len(curve["precisions"]), len(curve["recalls"]))

    def test_best_threshold_for_min_recall_contract(self):
        best = best_threshold_for_min_recall(self.y_true, self.y_prob, min_recall=0.75)
        self.assertIn("threshold", best)
        self.assertIn("precision", best)
        self.assertIn("recall", best)

    def test_cross_validate_precision_recall_shapes(self):
        model = build_logistic_pipeline(
            categorical_cols=self.cat_cols,
            numerical_cols=self.num_cols,
            max_iter=1000,
            random_state=42,
        )
        result = cross_validate_precision_recall(model, self.X, self.y, cv=5, random_state=42)

        self.assertEqual(len(result["precision_scores"]), 5)
        self.assertEqual(len(result["recall_scores"]), 5)
        self.assertEqual(len(result["f1_scores"]), 5)

    def test_fbeta_non_negative(self):
        f2 = evaluate_fbeta(self.y_true, self.y_pred, beta=2.0)
        self.assertGreaterEqual(f2, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
