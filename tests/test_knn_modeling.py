"""Unit tests for KNN modeling utilities."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.knn_modeling import (
    build_knn_classifier_pipeline,
    build_knn_regressor_pipeline,
    compare_classification_accuracy,
    cross_validate_knn_classifier,
    cross_validate_knn_regression,
    evaluate_knn_classification,
    evaluate_knn_regression,
    tune_knn_classifier_k,
)


class TestKnnModeling(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)

        n_cls = 260
        X_num_cls = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 45, size=n_cls),
                "items_count": rng.randint(1, 20, size=n_cls),
                "order_value": rng.uniform(15, 450, size=n_cls),
                "day_of_month": rng.randint(1, 31, size=n_cls),
            }
        )
        X_cat_cls = pd.DataFrame(
            {
                "zone": rng.choice(["Zone_A", "Zone_B", "Zone_C"], size=n_cls),
                "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n_cls),
                "peak_hour": rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n_cls),
            }
        )
        self.X_cls = pd.concat([X_num_cls, X_cat_cls], axis=1)

        signal = (
            0.06 * X_num_cls["distance_km"]
            + 0.05 * X_num_cls["items_count"]
            + 0.002 * X_num_cls["order_value"]
            + (X_cat_cls["peak_hour"] == "Evening").astype(int) * 0.4
            + rng.normal(0, 0.8, size=n_cls)
        )
        self.y_cls = (signal > np.median(signal)).astype(int)

        n_reg = 220
        self.X_reg = np.column_stack(
            [
                rng.uniform(0, 25, size=n_reg),
                rng.uniform(0, 1, size=n_reg),
                rng.uniform(10, 250, size=n_reg),
            ]
        )
        self.y_reg = (
            2.0 * np.sin(self.X_reg[:, 0] / 3.5)
            + 1.5 * self.X_reg[:, 1]
            + 0.02 * self.X_reg[:, 2]
            + rng.normal(0, 1.0, size=n_reg)
        )

    def test_build_knn_classifier_pipeline_has_steps(self):
        pipeline = build_knn_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_neighbors=5,
        )
        self.assertIn("preprocessor", pipeline.named_steps)
        self.assertIn("model", pipeline.named_steps)

    def test_evaluate_knn_classification_keys(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        metrics = evaluate_knn_classification(y_true, y_pred)
        self.assertEqual(set(metrics.keys()), {"accuracy", "precision", "recall", "f1"})

    def test_tune_knn_classifier_k_contract(self):
        pipeline = build_knn_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_neighbors=5,
        )
        result = tune_knn_classifier_k(
            pipeline,
            self.X_cls,
            self.y_cls,
            k_values=[1, 3, 5, 7],
            cv=3,
            random_state=42,
        )
        self.assertEqual(len(result["k_values"]), 4)
        self.assertEqual(len(result["mean_cv_scores"]), 4)
        self.assertEqual(len(result["mean_train_scores"]), 4)
        self.assertIn(result["best_k"], [1, 3, 5, 7])

    def test_cross_validate_knn_classifier_shapes(self):
        pipeline = build_knn_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_neighbors=5,
        )
        result = cross_validate_knn_classifier(pipeline, self.X_cls, self.y_cls, cv=4, random_state=42)
        self.assertEqual(len(result["accuracy_scores"]), 4)
        self.assertEqual(len(result["f1_scores"]), 4)

    def test_build_knn_regressor_pipeline_predicts(self):
        pipeline = build_knn_regressor_pipeline(n_neighbors=7, scale_features=True)
        X_train, X_test, y_train, _ = train_test_split(self.X_reg, self.y_reg, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        self.assertEqual(pred.shape[0], X_test.shape[0])

    def test_evaluate_knn_regression_keys(self):
        y_true = np.array([1.2, 2.0, 2.8, 3.1])
        y_pred = np.array([1.0, 2.2, 2.6, 3.0])
        metrics = evaluate_knn_regression(y_true, y_pred)
        self.assertEqual(set(metrics.keys()), {"mse", "rmse", "mae", "r2"})
        self.assertGreaterEqual(metrics["rmse"], 0.0)

    def test_compare_classification_accuracy_gain(self):
        result = compare_classification_accuracy(model_accuracy=0.82, baseline_accuracy=0.70)
        self.assertAlmostEqual(result["accuracy_gain"], 0.12, places=7)

    def test_cross_validate_knn_regression_shapes(self):
        pipeline = build_knn_regressor_pipeline(n_neighbors=5, scale_features=True)
        result = cross_validate_knn_regression(pipeline, self.X_reg, self.y_reg, cv=4, random_state=42)
        self.assertEqual(len(result["rmse_scores"]), 4)
        self.assertEqual(len(result["r2_scores"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)