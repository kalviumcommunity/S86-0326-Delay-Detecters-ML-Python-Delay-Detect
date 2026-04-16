"""Hyperparameter-tuning helpers using GridSearchCV and RandomizedSearchCV."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from scipy.stats import randint
from sklearn.metrics import classification_report
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import Config


def _infer_feature_groups(X) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature names from tabular input."""
    if hasattr(X, "select_dtypes") and hasattr(X, "columns"):
        numeric_cols = list(X.select_dtypes(include=["number"]).columns)
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        return numeric_cols, categorical_cols

    n_cols = X.shape[1]
    return [f"feature_{i}" for i in range(n_cols)], []


def build_knn_pipeline(X) -> Pipeline:
    """Create a leakage-safe KNN pipeline with fold-safe preprocessing."""
    numeric_cols, categorical_cols = _infer_feature_groups(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier()),
    ])


def run_knn_grid_search(
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 5,
    random_state: int = Config.RANDOM_STATE,
) -> GridSearchCV:
    """Run exhaustive KNN hyperparameter tuning via GridSearchCV."""
    pipeline = build_knn_pipeline(X_train)
    param_grid = {
        "knn__n_neighbors": list(range(1, 21)),
        "knn__weights": ["uniform", "distance"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def run_knn_random_search(
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 5,
    n_iter: int = 40,
    random_state: int = Config.RANDOM_STATE,
) -> RandomizedSearchCV:
    """Run randomized KNN tuning for larger search spaces or lower compute budgets."""
    pipeline = build_knn_pipeline(X_train)
    param_distributions = {
        "knn__n_neighbors": randint(1, 60),
        "knn__weights": ["uniform", "distance"],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def run_decision_tree_grid_search(
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 5,
    random_state: int = Config.RANDOM_STATE,
) -> GridSearchCV:
    """Run exhaustive DecisionTree hyperparameter tuning via GridSearchCV."""
    numeric_cols, categorical_cols = _infer_feature_groups(X_train)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=random_state)),
    ])
    param_grid = {
        "tree__max_depth": [2, 4, 6, 8, 10, None],
        "tree__min_samples_leaf": [1, 5, 10, 20],
        "tree__criterion": ["gini", "entropy"],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def summarize_search_results(search, top_n: int = 10) -> pd.DataFrame:
    """Return a tidy top-N summary table from search.cv_results_."""
    results_df = pd.DataFrame(search.cv_results_)

    preferred_cols = [
        c
        for c in [
            "mean_train_score",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
        if c in results_df.columns
    ]
    param_cols = [c for c in results_df.columns if c.startswith("param_")]

    view = results_df[param_cols + preferred_cols].sort_values("rank_test_score").head(top_n).copy()

    for col in ["mean_train_score", "mean_test_score", "std_test_score"]:
        if col in view.columns:
            view[col] = view[col].astype(float).round(4)

    return view.reset_index(drop=True)


def coarse_to_fine_knn_grids() -> Dict[str, Dict[str, list]]:
    """Return coarse and fine KNN grids for staged tuning."""
    return {
        "coarse": {
            "knn__n_neighbors": [1, 3, 5, 9, 15, 25, 35],
            "knn__weights": ["uniform", "distance"],
        },
        "fine": {
            "knn__n_neighbors": [7, 9, 11, 13, 15, 17],
            "knn__weights": ["uniform", "distance"],
        },
    }


def evaluate_best_on_test(search, X_test, y_test) -> Dict[str, object]:
    """Evaluate the fitted best estimator on test data exactly once."""
    best = search.best_estimator_
    y_pred = best.predict(X_test)

    score_name = "score"
    test_score = float(best.score(X_test, y_test))

    if hasattr(search, "scoring") and isinstance(search.scoring, str):
        score_name = search.scoring
        scorer = get_scorer(score_name)
        test_score = float(scorer(best, X_test, y_test))

    return {
        "test_score": test_score,
        "score_name": score_name,
        "classification_report": classification_report(y_test, y_pred),
    }
