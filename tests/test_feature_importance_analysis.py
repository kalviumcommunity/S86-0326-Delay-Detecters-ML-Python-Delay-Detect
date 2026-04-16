"""Unit tests for feature-importance analysis helpers."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_importance_analysis import (
    build_random_forest_classifier_pipeline,
    compare_importance_tables,
    evaluate_drop_impact,
    find_high_correlation_pairs,
    get_mdi_importance_table,
    get_permutation_importance_table,
    train_random_forest_pipeline,
)


class TestFeatureImportanceAnalysis(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        n = 280

        self.df = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 40, size=n),
                "items_count": rng.randint(1, 25, size=n),
                "order_value": rng.uniform(10, 500, size=n),
                "day_of_month": rng.randint(1, 31, size=n),
                "zone": rng.choice(["Zone_A", "Zone_B", "Zone_C", "Zone_D"], size=n),
                "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], size=n),
                "peak_hour": rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n),
            }
        )

        # Add a strongly correlated numeric feature to validate correlation detection.
        self.df["order_value_clone"] = self.df["order_value"] * 0.98 + rng.normal(0, 1.0, size=n)

        signal = (
            0.06 * self.df["distance_km"]
            + 0.04 * self.df["items_count"]
            + 0.002 * self.df["order_value"]
            + (self.df["peak_hour"] == "Evening").astype(int) * 0.55
            + rng.normal(0, 0.8, size=n)
        )
        self.y = (signal > np.median(signal)).astype(int)

        self.X = self.df[
            [
                "distance_km",
                "items_count",
                "order_value",
                "day_of_month",
                "zone",
                "day_of_week",
                "peak_hour",
            ]
        ]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )

    def test_random_forest_pipeline_trains(self):
        model = build_random_forest_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_estimators=100,
            random_state=42,
        )
        model = train_random_forest_pipeline(model, self.X_train, self.y_train)
        self.assertGreaterEqual(model.score(self.X_test, self.y_test), 0.5)

    def test_mdi_importance_contract(self):
        model = build_random_forest_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_estimators=80,
            random_state=42,
        )
        model = train_random_forest_pipeline(model, self.X_train, self.y_train)
        mdi = get_mdi_importance_table(model)
        self.assertIn("Feature", mdi.columns)
        self.assertIn("MDI_Importance", mdi.columns)
        self.assertAlmostEqual(float(mdi["MDI_Importance"].sum()), 1.0, places=6)

    def test_permutation_importance_contract(self):
        model = build_random_forest_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_estimators=80,
            random_state=42,
        )
        model = train_random_forest_pipeline(model, self.X_train, self.y_train)
        perm = get_permutation_importance_table(
            model,
            self.X_test,
            self.y_test,
            scoring="accuracy",
            n_repeats=5,
            random_state=42,
        )
        self.assertIn("Feature", perm.columns)
        self.assertIn("Permutation_Importance", perm.columns)
        self.assertEqual(len(perm), self.X_test.shape[1])
        self.assertLessEqual(len(perm), len(get_mdi_importance_table(model)))

    def test_compare_importance_tables_contract(self):
        model = build_random_forest_classifier_pipeline(
            categorical_cols=["zone", "day_of_week", "peak_hour"],
            numerical_cols=["distance_km", "items_count", "order_value", "day_of_month"],
            n_estimators=80,
            random_state=42,
        )
        model = train_random_forest_pipeline(model, self.X_train, self.y_train)
        mdi = get_mdi_importance_table(model)
        perm = get_permutation_importance_table(model, self.X_test, self.y_test, n_repeats=5)
        merged = compare_importance_tables(mdi, perm)
        self.assertIn("MDI_Rank", merged.columns)
        self.assertIn("Permutation_Rank", merged.columns)
        self.assertIn("Rank_Delta_MDI_minus_Permutation", merged.columns)

    def test_find_high_correlation_pairs_detects_strong_pair(self):
        corr_pairs = find_high_correlation_pairs(
            self.df[["order_value", "order_value_clone", "distance_km"]],
            threshold=0.8,
        )
        self.assertGreaterEqual(len(corr_pairs), 1)
        top_pair = {corr_pairs.iloc[0]["Feature_A"], corr_pairs.iloc[0]["Feature_B"]}
        self.assertEqual(top_pair, {"order_value", "order_value_clone"})

    def test_drop_impact_contract(self):
        impact = evaluate_drop_impact(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            drop_raw_features=["day_of_month"],
            random_state=42,
        )
        self.assertIn("baseline_accuracy", impact)
        self.assertIn("dropped_accuracy", impact)
        self.assertIn("accuracy_delta", impact)
        self.assertIn("dropped_features", impact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
