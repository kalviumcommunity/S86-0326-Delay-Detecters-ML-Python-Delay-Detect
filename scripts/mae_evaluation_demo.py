"""Lesson 5.23 demo: evaluating regression models with MAE.

This script compares a mean baseline vs linear regression and reports:
- MAE in target units
- MAE improvement over baseline
- MAE as percentage of mean target
- MAE cross-validation stability
- residual diagnostics

Run:
    python scripts/mae_evaluation_demo.py
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
from src.mae_evaluation import (
    compare_model_vs_baseline_mae,
    cross_validate_mae,
    evaluate_regression_with_mae,
    mae_as_percentage_of_mean_target,
    residual_diagnostics,
)
from src.regression import build_linear_regression_pipeline, train_regression_baseline
from src.data_preprocessing import clean_data, load_data


def make_continuous_target(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Construct synthetic continuous target from project features."""
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

    feature_cols = Config.NUMERICAL_COLS
    X = df[feature_cols].copy()
    y = make_continuous_target(df, seed=Config.RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    print("\n1) Baseline (mean) evaluation")
    baseline = train_regression_baseline(X_train, y_train, strategy="mean")
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_regression_with_mae(y_test, baseline_pred)

    print("\n2) Linear Regression evaluation")
    model = build_linear_regression_pipeline(scale_features=True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_metrics = evaluate_regression_with_mae(y_test, model_pred)

    mae_comp = compare_model_vs_baseline_mae(
        model_mae=model_metrics["mae"],
        baseline_mae=baseline_metrics["mae"],
    )

    mae_pct = mae_as_percentage_of_mean_target(model_metrics["mae"], y_test)
    residuals = residual_diagnostics(y_test, model_pred)

    print("Baseline metrics:")
    print({k: round(v, 4) for k, v in baseline_metrics.items()})
    print("Model metrics:")
    print({k: round(v, 4) for k, v in model_metrics.items()})
    print("MAE comparison:")
    print({k: round(v, 4) for k, v in mae_comp.items()})
    print(f"Model MAE as % of mean target: {mae_pct:.2f}%")
    print("Residual diagnostics:")
    print({k: round(v, 4) for k, v in residuals.items()})

    print("\n3) Cross-validation with MAE")
    cv_result = cross_validate_mae(model, X_train, y_train, cv=5)
    print("CV MAE per fold:", np.round(cv_result["scores"], 4).tolist())
    print(f"Mean CV MAE: {cv_result['mean']:.4f} +/- {cv_result['std']:.4f}")


if __name__ == "__main__":
    main()
