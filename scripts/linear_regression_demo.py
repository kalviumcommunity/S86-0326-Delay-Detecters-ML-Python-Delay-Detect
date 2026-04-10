"""Lesson 5.22 demo: baseline vs linear regression.

This script creates a synthetic continuous target from project features,
then compares DummyRegressor baseline and LinearRegression on the same split.

Run:
    python scripts/linear_regression_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running as a direct script from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data
from src.regression import (
    build_linear_regression_pipeline,
    compare_model_to_baseline,
    cross_validate_r2,
    evaluate_regression_model,
    get_linear_coefficients,
    train_regression_baseline,
)


def make_continuous_target(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Construct a realistic continuous delivery-time target for regression demo."""
    rng = np.random.RandomState(seed)

    # Continuous target in minutes.
    y = (
        12.0
        + 1.35 * df["distance_km"]
        + 0.95 * df["items_count"]
        + 0.015 * df["order_value"]
        + 0.25 * df["day_of_month"]
        + rng.normal(0, 4.0, size=len(df))
    )
    return y


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

    print("\n1) Training mean baseline...")
    baseline = train_regression_baseline(X_train, y_train, strategy="mean")
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_regression_model(y_test, baseline_pred)

    print("\n2) Training linear regression pipeline...")
    pipeline = build_linear_regression_pipeline(scale_features=True)
    pipeline.fit(X_train, y_train)
    model_pred = pipeline.predict(X_test)
    model_metrics = evaluate_regression_model(y_test, model_pred)

    print("\n3) Comparing against baseline...")
    improvement = compare_model_to_baseline(model_metrics, baseline_metrics)

    print("Baseline metrics:")
    print({k: round(v, 4) for k, v in baseline_metrics.items()})
    print("Linear Regression metrics:")
    print({k: round(v, 4) for k, v in model_metrics.items()})
    print("Improvements:")
    print({k: round(v, 4) for k, v in improvement.items()})

    print("\n4) Cross-validation stability (R2)...")
    cv_result = cross_validate_r2(pipeline, X_train, y_train, cv=5)
    print("CV R2 scores:", np.round(cv_result["scores"], 4).tolist())
    print(f"Mean CV R2: {cv_result['mean']:.4f} +/- {cv_result['std']:.4f}")

    print("\n5) Coefficients...")
    coef_df = get_linear_coefficients(pipeline, feature_cols)
    intercept = float(pipeline.named_steps["model"].intercept_)
    print(f"Intercept: {intercept:.4f}")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
