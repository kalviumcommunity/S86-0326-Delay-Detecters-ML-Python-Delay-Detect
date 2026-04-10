"""Unit tests for confusion-matrix focused evaluation helpers."""

import unittest

import numpy as np

from src.confusion_matrix_evaluation import (
    compute_confusion_matrix,
    confusion_metrics_from_cells,
    extract_binary_confusion_cells,
    most_confused_class_pairs,
    normalize_confusion_matrix,
    threshold_confusion_table,
)


class TestConfusionMatrixEvaluation(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0])
        self.y_prob = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8, 0.7, 0.3, 0.45, 0.2])

    def test_compute_confusion_matrix_shape_binary(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.shape, (2, 2))

    def test_extract_binary_confusion_cells_keys(self):
        cells = extract_binary_confusion_cells(self.y_true, self.y_pred)
        self.assertEqual(set(cells.keys()), {"tn", "fp", "fn", "tp"})

    def test_confusion_metrics_from_cells_contract(self):
        metrics = confusion_metrics_from_cells(tp=4, fp=1, fn=2, tn=3)
        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_normalize_confusion_matrix_true_row_sums(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred)
        cm_norm = normalize_confusion_matrix(cm, normalize="true")
        row_sums = cm_norm.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums)))

    def test_threshold_confusion_table_length(self):
        rows = threshold_confusion_table(self.y_true, self.y_prob, thresholds=[0.3, 0.5, 0.7])
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows[0].keys()), {"threshold", "tp", "fp", "fn", "tn"})

    def test_most_confused_class_pairs_returns_top_k(self):
        cm = np.array(
            [
                [8, 2, 0],
                [1, 6, 3],
                [0, 4, 5],
            ]
        )
        labels = ["A", "B", "C"]
        top = most_confused_class_pairs(cm, labels, top_k=2)
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0]["count"], top[1]["count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)