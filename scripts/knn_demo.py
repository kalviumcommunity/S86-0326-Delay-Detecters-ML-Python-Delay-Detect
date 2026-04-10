"""Lesson 5.30 demo: training K-Nearest Neighbors (KNN) models.

Run:
    python scripts/knn_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.knn_modeling import (
    build_knn_classifier_pipeline,
    build_knn_regressor_pipeline,
    compare_classification_accuracy,
    cross_validate_knn_classifier,
    cross_validate_knn_regression,
    evaluate_knn_classification,
    evaluate_knn_regression,
    tune_knn_classifier_k,
)


def classification_demo() -> None:
    print("=== KNN Classification Demo (Project Data) ===")
    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    baseline_metrics = evaluate_knn_classification(y_test, y_base)

    pipeline = build_knn_classifier_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        n_neighbors=5,
        metric="euclidean",
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    model_metrics = evaluate_knn_classification(y_test, y_pred)

    print("Baseline metrics:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    print("KNN metrics (K=5):", {k: round(v, 4) for k, v in model_metrics.items()})

    gain = compare_classification_accuracy(
        model_accuracy=model_metrics["accuracy"],
        baseline_accuracy=baseline_metrics["accuracy"],
    )
    printable_gain = {
        k: ("inf" if np.isinf(v) else round(v, 4))
        for k, v in gain.items()
    }
    print("Accuracy gain vs baseline:", printable_gain)

    print("\nK selection using CV (odd K from 1 to 21)")
    k_values = list(range(1, 22, 2))
    tuning = tune_knn_classifier_k(
        pipeline,
        X_train,
        y_train,
        k_values=k_values,
        cv=5,
        random_state=Config.RANDOM_STATE,
    )
    print("Best K:", tuning["best_k"])
    print(f"Best CV accuracy: {tuning['best_cv_score']:.4f}")
    print("K -> CV accuracy:")
    for k, score in zip(tuning["k_values"], tuning["mean_cv_scores"]):
        print(f"  K={k:2d}: {score:.4f}")

    best_model = pipeline.set_params(model__n_neighbors=tuning["best_k"])
    cv_result = cross_validate_knn_classifier(
        best_model,
        X_train,
        y_train,
        cv=5,
        random_state=Config.RANDOM_STATE,
    )
    print("CV accuracy scores:", np.round(cv_result["accuracy_scores"], 4).tolist())
    print(f"Mean CV accuracy: {cv_result['accuracy_mean']:.4f} +/- {cv_result['accuracy_std']:.4f}")
    print("CV F1 scores:", np.round(cv_result["f1_scores"], 4).tolist())
    print(f"Mean CV F1: {cv_result['f1_mean']:.4f} +/- {cv_result['f1_std']:.4f}")


def regression_demo() -> None:
    print("\n=== KNN Regression Demo (Synthetic Data) ===")
    rng = np.random.RandomState(42)
    n = 240

    X = np.column_stack(
        [
            rng.uniform(0, 20, size=n),
            rng.uniform(0, 1, size=n),
            rng.uniform(10, 200, size=n),
        ]
    )
    noise = rng.normal(0, 1.2, size=n)
    y = 2.5 * np.sin(X[:, 0] / 3.0) + 1.8 * X[:, 1] + 0.03 * X[:, 2] + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    baseline_metrics = evaluate_knn_regression(y_test, y_base)

    reg_pipeline = build_knn_regressor_pipeline(
        n_neighbors=7,
        metric="euclidean",
        scale_features=True,
    )
    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)
    model_metrics = evaluate_knn_regression(y_test, y_pred)

    print("Baseline regression metrics:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    print("KNN regression metrics (K=7):", {k: round(v, 4) for k, v in model_metrics.items()})
    print(f"R2 gain vs baseline: {model_metrics['r2'] - baseline_metrics['r2']:.4f}")

    cv_result = cross_validate_knn_regression(reg_pipeline, X_train, y_train, cv=5, random_state=42)
    print("CV RMSE scores:", np.round(cv_result["rmse_scores"], 4).tolist())
    print(f"Mean CV RMSE: {cv_result['rmse_mean']:.4f} +/- {cv_result['rmse_std']:.4f}")
    print("CV R2 scores:", np.round(cv_result["r2_scores"], 4).tolist())
    print(f"Mean CV R2: {cv_result['r2_mean']:.4f} +/- {cv_result['r2_std']:.4f}")


def main() -> None:
    classification_demo()
    regression_demo()


if __name__ == "__main__":
    main()