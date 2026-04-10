"""Linear regression training and evaluation utilities.

Provides a clean workflow for comparing a DummyRegressor baseline against a
LinearRegression model, including cross-validation and coefficient inspection.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_regression_baseline(
    X_train,
    y_train,
    strategy: str = "mean",
) -> DummyRegressor:
    """Fit a baseline regressor on training data only."""
    baseline = DummyRegressor(strategy=strategy)
    baseline.fit(X_train, y_train)
    return baseline


def build_linear_regression_pipeline(scale_features: bool = True) -> Pipeline:
    """Build a leakage-safe linear regression pipeline."""
    if scale_features:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    return Pipeline([
        ("model", LinearRegression()),
    ])


def evaluate_regression_model(y_true, y_pred) -> Dict[str, float]:
    """Return standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_model_to_baseline(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute metric deltas where lower-is-better and higher-is-better are handled."""
    return {
        "mse_improvement": baseline_metrics["mse"] - model_metrics["mse"],
        "rmse_improvement": baseline_metrics["rmse"] - model_metrics["rmse"],
        "mae_improvement": baseline_metrics["mae"] - model_metrics["mae"],
        "r2_improvement": model_metrics["r2"] - baseline_metrics["r2"],
    }


def cross_validate_r2(model_pipeline: Pipeline, X_train, y_train, cv: int = 5) -> Dict[str, object]:
    """Run CV and return fold R2 values with mean/std summary."""
    scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="r2")
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def get_linear_coefficients(model_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract coefficients from a fitted linear regression pipeline."""
    model = model_pipeline.named_steps["model"]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "abs_coefficient": np.abs(model.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df
