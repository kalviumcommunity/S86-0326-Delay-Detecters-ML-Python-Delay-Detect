"""MAE-focused utilities for regression model evaluation.

This module standardizes MAE-centric evaluation and reporting patterns,
including baseline comparison and cross-validation handling.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_regression_with_mae(y_true, y_pred) -> Dict[str, float]:
    """Compute MAE-first regression metrics on predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def mae_as_percentage_of_mean_target(mae: float, y_true) -> float:
    """Express MAE relative to average target magnitude (percentage)."""
    mean_target = float(np.mean(y_true))
    if mean_target == 0:
        raise ValueError("Mean target is zero; percentage interpretation is undefined")
    return (mae / mean_target) * 100.0


def compare_model_vs_baseline_mae(model_mae: float, baseline_mae: float) -> Dict[str, float]:
    """Return absolute and percent MAE improvement over baseline."""
    improvement = baseline_mae - model_mae
    if baseline_mae == 0:
        pct_improvement = 0.0
    else:
        pct_improvement = (improvement / baseline_mae) * 100.0

    return {
        "mae_improvement": improvement,
        "mae_improvement_pct": pct_improvement,
    }


def cross_validate_mae(model, X_train, y_train, cv: int = 5) -> Dict[str, object]:
    """Run K-fold CV using MAE and return fold scores with summary.

    scikit-learn returns negative MAE for consistency with "higher-is-better"
    scorers. This function flips signs to report conventional positive MAE.
    """
    neg_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    mae_scores = -neg_scores

    return {
        "scores": mae_scores,
        "mean": float(mae_scores.mean()),
        "std": float(mae_scores.std()),
    }


def residual_diagnostics(y_true, y_pred) -> Dict[str, float]:
    """Provide simple residual summary to detect directional bias."""
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_std": float(np.std(residuals)),
    }
