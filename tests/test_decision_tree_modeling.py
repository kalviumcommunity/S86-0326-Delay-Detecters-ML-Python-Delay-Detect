"""Unit tests for decision tree modeling utilities."""

import unittest

import numpy as np
import pandas as pd

from src.decision_tree_modeling import (
    build_decision_tree_classifier_pipeline,
    build_decision_tree_regressor,
    build_feature_importance_frame,
    compare_classification_accuracy,
    cross_validate_decision_tree_classifier,
    cross_validate_decision_tree_regression,
    evaluate_decision_tree_classification,
    evaluate_decision_tree_regression,
    get_tree_feature_names,
    tune_decision_tree_classifier_depth,
)


class TestDecisionTreeModeling(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)

        n_samples = 240
        numeric = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 45, size=n_samples),
                "items_count": rng.randint(1, 20, size=n_samples),
                "order_value": rng.uniform(15, 450, size=n_samples),
                "day_of_month": rng.randint(1, 31, size=n_samples),
            }
        )
        categorical = pd.DataFrame(
            {
                "zone": rng.choice(["Zone_A", "Zone_B", "Zone_C"], size=n_samples),
                "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n_samples),
                "peak_hour": rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n_samples),
            }
        )
        self.X_cls = pd.concat([numeric, categorical], axis=1)

        signal = (
            0.07 * numeric["distance_km"]
            + 0.05 * numeric["items_count"]
            + 0.002 * numeric["order_value"]
            + (categorical["peak_hour"] == "Evening").astype(int) * 0.5
            + rng.normal(0, 0.9, size=n_samples)
        )
        self.y_cls = (signal > np.median(signal)).astype(int)

        self.X_reg = np.column_stack(
            [
                rng.uniform(0, 25, size=n_samples),
                rng.uniform(0, 1, size=n_samples),
                rng.uniform(10, 250, size=n_samples),
            ]
        )
        self.y_reg = (
            2.2 * np.sin(self.X_reg[:, 0] / 3.5)
            + 1.4 * self.X_reg[:, 1]
            + 0.02 * self.X_reg[:, 2]
            + rng.normal(0, 1.0, size=n_samples)
        )

    def test_build_classifier_pipeline_has_steps(self):
        pipeline = build_decision_tree_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            max_depth=4,
        )
        self.assertIn("preprocessor", pipeline.named_steps)
        self.assertIn("model", pipeline.named_steps)

    def test_evaluate_classification_keys(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        metrics = evaluate_decision_tree_classification(y_true, y_pred)
        self.assertEqual(set(metrics.keys()), {"accuracy", "precision", "recall", "f1"})

    def test_depth_tuning_contract(self):
        pipeline = build_decision_tree_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            max_depth=4,
        )
        result = tune_decision_tree_classifier_depth(
            pipeline,
            self.X_cls,
            self.y_cls,
            depth_values=[1, 2, 3, 4],
            cv=3,
            random_state=42,
        )
        self.assertEqual(len(result["depth_values"]), 4)
        self.assertEqual(len(result["mean_cv_scores"]), 4)
        self.assertEqual(len(result["mean_train_scores"]), 4)
        self.assertIn(result["best_depth"], [1, 2, 3, 4])

    def test_cross_validate_classifier_shapes(self):
        pipeline = build_decision_tree_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            max_depth=4,
        )
        result = cross_validate_decision_tree_classifier(pipeline, self.X_cls, self.y_cls, cv=4, random_state=42)
        self.assertEqual(len(result["accuracy_scores"]), 4)
        self.assertEqual(len(result["f1_scores"]), 4)

    def test_feature_names_and_importance_table(self):
        pipeline = build_decision_tree_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            max_depth=4,
        )
        pipeline.fit(self.X_cls, self.y_cls)
        feature_names = get_tree_feature_names(pipeline)
        importance_df = build_feature_importance_frame(pipeline)
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(set(importance_df.columns), {"Feature", "Importance"})
        self.assertEqual(len(importance_df), len(feature_names))

    def test_compare_classification_accuracy_gain(self):
        result = compare_classification_accuracy(model_accuracy=0.83, baseline_accuracy=0.71)
        self.assertAlmostEqual(result["accuracy_gain"], 0.12, places=7)

    def test_regressor_metrics_keys(self):
        y_true = np.array([1.1, 2.0, 3.1, 4.2])
        y_pred = np.array([1.0, 2.2, 2.8, 4.0])
        metrics = evaluate_decision_tree_regression(y_true, y_pred)
        self.assertEqual(set(metrics.keys()), {"mse", "rmse", "mae", "r2"})
        self.assertGreaterEqual(metrics["rmse"], 0.0)

    def test_cross_validate_regression_shapes(self):
        model = build_decision_tree_regressor(max_depth=4, min_samples_leaf=5, random_state=42)
        result = cross_validate_decision_tree_regression(model, self.X_reg, self.y_reg, cv=4, random_state=42)
        self.assertEqual(len(result["rmse_scores"]), 4)
        self.assertEqual(len(result["r2_scores"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)