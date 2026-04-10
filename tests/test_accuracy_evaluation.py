"""Unit tests for accuracy-focused evaluation helpers."""

import unittest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.accuracy_evaluation import (
    compare_with_baseline_accuracy,
    confusion_matrix_breakdown,
    cross_validate_accuracy,
    evaluate_accuracy_metrics,
    majority_class_baseline_accuracy,
)
from src.logistic_classification import build_logistic_pipeline


class TestAccuracyEvaluation(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

        rng = np.random.RandomState(42)
        n = 200
        distance = rng.uniform(1, 50, size=n)
        items = rng.randint(1, 20, size=n)
        order_value = rng.uniform(10, 500, size=n)
        day_of_month = rng.randint(1, 31, size=n)

        zone = rng.choice(["Zone_A", "Zone_B", "Zone_C"], size=n)
        dow = rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n)
        peak = rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n)

        signal = 0.06 * distance + 0.05 * items + 0.004 * order_value + (peak == "Evening") * 0.35 + rng.normal(0, 0.8, size=n)
        y = (signal > np.median(signal)).astype(int)

        self.X = {
            "distance_km": distance,
            "items_count": items,
            "order_value": order_value,
            "day_of_month": day_of_month,
            "zone": zone,
            "day_of_week": dow,
            "peak_hour": peak,
        }
        import pandas as pd
        self.X = pd.DataFrame(self.X)
        self.y = y

    def test_evaluate_accuracy_metrics_keys(self):
        metrics = evaluate_accuracy_metrics(self.y_true, self.y_pred)
        self.assertEqual(set(metrics.keys()), {"accuracy", "balanced_accuracy"})

    def test_confusion_breakdown_sums_to_total(self):
        cells = confusion_matrix_breakdown(self.y_true, self.y_pred)
        self.assertEqual(cells["tn"] + cells["fp"] + cells["fn"] + cells["tp"], len(self.y_true))

    def test_majority_class_baseline_accuracy(self):
        y = np.array([0, 0, 0, 1, 1])
        self.assertAlmostEqual(majority_class_baseline_accuracy(y), 0.6)

    def test_compare_with_baseline_accuracy(self):
        gain = compare_with_baseline_accuracy(model_acc=0.72, baseline_acc=0.60)
        self.assertAlmostEqual(gain["accuracy_gain"], 0.12, places=7)
        self.assertAlmostEqual(gain["relative_gain_pct"], 20.0, places=7)

    def test_cross_validate_accuracy_output_shapes(self):
        model = build_logistic_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            max_iter=1000,
            random_state=42,
        )

        result = cross_validate_accuracy(model, self.X, self.y, cv=5, random_state=42)
        self.assertEqual(len(result["accuracy_scores"]), 5)
        self.assertEqual(len(result["balanced_accuracy_scores"]), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
