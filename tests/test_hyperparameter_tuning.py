"""Unit tests for hyperparameter tuning helpers."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.hyperparameter_tuning import (
    coarse_to_fine_knn_grids,
    evaluate_best_on_test,
    run_decision_tree_grid_search,
    run_knn_grid_search,
    run_knn_random_search,
    summarize_search_results,
)


class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        n = 260

        df = pd.DataFrame(
            {
                "distance_km": rng.uniform(1, 30, size=n),
                "items_count": rng.randint(1, 20, size=n),
                "order_value": rng.uniform(20, 400, size=n),
                "day_of_month": rng.randint(1, 31, size=n),
                "zone": rng.choice(["A", "B", "C"], size=n),
                "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n),
                "peak_hour": rng.choice(["Morning", "Afternoon", "Evening", "Night"], size=n),
            }
        )

        signal = (
            0.05 * df["distance_km"]
            + 0.03 * df["items_count"]
            + 0.001 * df["order_value"]
            + (df["peak_hour"] == "Evening").astype(int) * 0.6
            + rng.normal(0, 0.7, size=n)
        )
        y = (signal > np.median(signal)).astype(int)

        X = df[
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
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    def test_knn_grid_search_runs(self):
        search = run_knn_grid_search(self.X_train, self.y_train, scoring="f1", cv=3)
        self.assertTrue(hasattr(search, "best_params_"))
        self.assertIn("knn__n_neighbors", search.best_params_)
        self.assertIn("knn__weights", search.best_params_)

    def test_knn_random_search_runs(self):
        search = run_knn_random_search(self.X_train, self.y_train, scoring="f1", cv=3, n_iter=8)
        self.assertTrue(hasattr(search, "best_params_"))
        self.assertIn("knn__n_neighbors", search.best_params_)

    def test_decision_tree_grid_search_runs(self):
        search = run_decision_tree_grid_search(self.X_train, self.y_train, scoring="f1", cv=3)
        self.assertTrue(hasattr(search, "best_score_"))
        self.assertIn("tree__max_depth", search.best_params_)

    def test_summary_table_contract(self):
        search = run_knn_grid_search(self.X_train, self.y_train, scoring="f1", cv=3)
        summary = summarize_search_results(search, top_n=5)
        self.assertEqual(len(summary), 5)
        self.assertIn("rank_test_score", summary.columns)
        self.assertIn("mean_test_score", summary.columns)

    def test_test_evaluation_contract(self):
        search = run_knn_grid_search(self.X_train, self.y_train, scoring="f1", cv=3)
        final_eval = evaluate_best_on_test(search, self.X_test, self.y_test)
        self.assertIn("test_score", final_eval)
        self.assertIn("score_name", final_eval)
        self.assertIn("classification_report", final_eval)

    def test_coarse_to_fine_grids_contract(self):
        grids = coarse_to_fine_knn_grids()
        self.assertIn("coarse", grids)
        self.assertIn("fine", grids)
        self.assertIn("knn__n_neighbors", grids["coarse"])
        self.assertIn("knn__n_neighbors", grids["fine"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
