"""Baseline model utilities for classification and regression tasks.

Baselines provide an honest minimum performance reference that every trained
model must beat.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def train_classification_baseline(
    X_train,
    y_train,
    strategy: str = "most_frequent",
    random_state: int = 42,
) -> DummyClassifier:
    """Fit a dummy classifier baseline on training labels only."""
    model = DummyClassifier(strategy=strategy, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_classification_baseline(
    baseline: DummyClassifier,
    X_test,
    y_test,
) -> Dict[str, float | None]:
    """Evaluate a fitted classification baseline on held-out test data."""
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length")

    y_pred = baseline.predict(X_test)
    result: Dict[str, float | None] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": None,
    }

    if len(np.unique(y_test)) == 2:
        try:
            y_proba = baseline.predict_proba(X_test)[:, 1]
            result["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            result["roc_auc"] = None

    return result


def train_regression_baseline(
    X_train,
    y_train,
    strategy: str = "mean",
) -> DummyRegressor:
    """Fit a dummy regressor baseline using mean/median strategy."""
    model = DummyRegressor(strategy=strategy)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_baseline(
    baseline: DummyRegressor,
    X_test,
    y_test,
) -> Dict[str, float]:
    """Evaluate a fitted regression baseline on held-out test data."""
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length")

    y_pred = baseline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def compare_to_baseline(
    model_metrics: Dict[str, float | None],
    baseline_metrics: Dict[str, float | None],
) -> Dict[str, float | None]:
    """Compute metric-wise model improvement relative to baseline."""
    improvements: Dict[str, float | None] = {}
    for metric_name, baseline_value in baseline_metrics.items():
        model_value = model_metrics.get(metric_name)
        if model_value is None or baseline_value is None:
            improvements[metric_name] = None
        else:
            improvements[metric_name] = model_value - baseline_value
    return improvements
