"""Unit tests for bias-variance diagnostics."""

import unittest

import numpy as np

from src.bias_variance import (
    decompose_expected_error,
    diagnose_train_test_behavior,
    interpret_learning_curve,
    summarize_cv_stability,
)


class TestBiasVariance(unittest.TestCase):
    def test_high_bias_signature(self):
        result = diagnose_train_test_behavior(0.62, 0.60, good_score_threshold=0.8, large_gap_threshold=0.1)
        self.assertEqual(result["diagnosis"], "high_bias")

    def test_high_variance_signature(self):
        result = diagnose_train_test_behavior(0.98, 0.74, good_score_threshold=0.8, large_gap_threshold=0.1)
        self.assertEqual(result["diagnosis"], "high_variance")

    def test_good_fit_signature(self):
        result = diagnose_train_test_behavior(0.89, 0.86, good_score_threshold=0.8, large_gap_threshold=0.1)
        self.assertEqual(result["diagnosis"], "good_fit")

    def test_cv_stability_labels(self):
        stable = summarize_cv_stability(np.array([0.82, 0.81, 0.83, 0.82, 0.82]), high_variability_threshold=0.03)
        unstable = summarize_cv_stability(np.array([0.92, 0.70, 0.88, 0.76, 0.91]), high_variability_threshold=0.03)
        self.assertEqual(stable["stability"], "stable")
        self.assertEqual(unstable["stability"], "unstable")

    def test_learning_curve_interpretation(self):
        train_scores = np.array(
            [
                [1.00, 1.00, 1.00],
                [0.99, 0.98, 0.99],
                [0.98, 0.98, 0.97],
            ]
        )
        val_scores = np.array(
            [
                [0.70, 0.72, 0.71],
                [0.75, 0.76, 0.74],
                [0.78, 0.77, 0.79],
            ]
        )
        result = interpret_learning_curve(train_scores, val_scores, good_score_threshold=0.8, large_gap_threshold=0.1)
        self.assertEqual(result["diagnosis"], "high_variance")

    def test_error_decomposition(self):
        total = decompose_expected_error(0.12, 0.08, 0.05)
        self.assertAlmostEqual(total, 0.25, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
