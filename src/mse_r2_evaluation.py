"""MSE and R2 focused utilities for regression evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_regression_with_mse_r2(y_true, y_pred) -> Dict[str, float]:
    """Compute MSE, RMSE, and R2 for predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def compare_model_vs_baseline_mse_r2(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Return metric deltas versus baseline.

    For MSE and RMSE, positive values indicate improvement (reduction).
    For R2, positive values indicate gain in explained variance.
    """
    return {
        "mse_reduction": baseline_metrics["mse"] - model_metrics["mse"],
        "rmse_reduction": baseline_metrics["rmse"] - model_metrics["rmse"],
        "r2_gain": model_metrics["r2"] - baseline_metrics["r2"],
    }


def cross_validate_r2(model, X_train, y_train, cv: int = 5) -> Dict[str, object]:
    """Cross-validate R2 (higher is better, no sign flip)."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def cross_validate_mse_rmse(model, X_train, y_train, cv: int = 5) -> Dict[str, object]:
    """Cross-validate MSE/RMSE with correct sign handling for scikit-learn scorers."""
    neg_mse = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    mse_scores = -neg_mse
    rmse_scores = np.sqrt(mse_scores)

    return {
        "mse_scores": mse_scores,
        "rmse_scores": rmse_scores,
        "mse_mean": float(mse_scores.mean()),
        "mse_std": float(mse_scores.std()),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
    }
