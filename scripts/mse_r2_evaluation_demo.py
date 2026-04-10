"""Lesson 5.24 demo: evaluate regression with MSE and R2.

Compares mean baseline and linear regression on:
- MSE
- RMSE
- R2
- cross-validated R2 and RMSE stability

Run:
    python scripts/mse_r2_evaluation_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data
from src.mse_r2_evaluation import (
    compare_model_vs_baseline_mse_r2,
    cross_validate_mse_rmse,
    cross_validate_r2,
    evaluate_regression_with_mse_r2,
)
from src.regression import build_linear_regression_pipeline, train_regression_baseline


def make_continuous_target(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Construct synthetic continuous target for regression evaluation."""
    rng = np.random.RandomState(seed)
    return (
        12.0
        + 1.35 * df["distance_km"]
        + 0.95 * df["items_count"]
        + 0.015 * df["order_value"]
        + 0.25 * df["day_of_month"]
        + rng.normal(0, 4.0, size=len(df))
    )


def main() -> None:
    print("Loading project data...")
    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df)

    X = df[Config.NUMERICAL_COLS].copy()
    y = make_continuous_target(df, seed=Config.RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    print("\n1) Baseline (mean)")
    baseline = train_regression_baseline(X_train, y_train, strategy="mean")
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_regression_with_mse_r2(y_test, baseline_pred)

    print("\n2) Linear Regression")
    model = build_linear_regression_pipeline(scale_features=True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_metrics = evaluate_regression_with_mse_r2(y_test, model_pred)

    print("\n3) Baseline comparison")
    comparison = compare_model_vs_baseline_mse_r2(model_metrics, baseline_metrics)

    print("Baseline metrics:")
    print({k: round(v, 4) for k, v in baseline_metrics.items()})
    print("Model metrics:")
    print({k: round(v, 4) for k, v in model_metrics.items()})
    print("Comparison:")
    print({k: round(v, 4) for k, v in comparison.items()})

    print("\n4) Cross-validation stability")
    cv_r2 = cross_validate_r2(model, X_train, y_train, cv=5)
    cv_err = cross_validate_mse_rmse(model, X_train, y_train, cv=5)

    print("CV R2 scores:", np.round(cv_r2["scores"], 4).tolist())
    print(f"Mean CV R2: {cv_r2['mean']:.4f} +/- {cv_r2['std']:.4f}")
    print("CV RMSE scores:", np.round(cv_err["rmse_scores"], 4).tolist())
    print(f"Mean CV RMSE: {cv_err['rmse_mean']:.4f} +/- {cv_err['rmse_std']:.4f}")


if __name__ == "__main__":
    main()
