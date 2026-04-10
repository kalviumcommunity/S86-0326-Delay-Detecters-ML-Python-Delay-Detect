"""Lesson 5.26 demo: evaluating classification models using accuracy.

Run:
    python scripts/accuracy_evaluation_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.accuracy_evaluation import (
    compare_with_baseline_accuracy,
    confusion_matrix_breakdown,
    cross_validate_accuracy,
    evaluate_accuracy_metrics,
    majority_class_baseline_accuracy,
)
from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.logistic_classification import build_logistic_pipeline


def main() -> None:
    print("Loading and splitting project data...")
    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    class_dist = y_test.value_counts(normalize=True).to_dict()
    print("Test class distribution:", {int(k): round(v, 4) for k, v in class_dist.items()})

    print("\n1) Majority baseline")
    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_accuracy_metrics(y_test, baseline_pred)

    expected_baseline_acc = majority_class_baseline_accuracy(y_test)
    print("Baseline metrics:")
    print({k: round(v, 4) for k, v in baseline_metrics.items()})
    print(f"Expected majority baseline accuracy from class ratio: {expected_baseline_acc:.4f}")

    print("\n2) Logistic Regression model")
    model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_metrics = evaluate_accuracy_metrics(y_test, model_pred)

    print("Model metrics:")
    print({k: round(v, 4) for k, v in model_metrics.items()})

    gain = compare_with_baseline_accuracy(
        model_acc=model_metrics["accuracy"],
        baseline_acc=baseline_metrics["accuracy"],
    )
    print("Accuracy gain over baseline:")
    print({k: round(v, 4) for k, v in gain.items()})

    print("\n3) Confusion matrix breakdown (model)")
    cm = confusion_matrix_breakdown(y_test, model_pred)
    print(cm)

    print("\nClassification report (model):")
    print(classification_report(y_test, model_pred, zero_division=0))

    print("\n4) Cross-validation stability")
    cv_result = cross_validate_accuracy(model, X_train, y_train, cv=5, random_state=Config.RANDOM_STATE)
    print("CV accuracy scores:", np.round(cv_result["accuracy_scores"], 4).tolist())
    print(f"Mean CV accuracy: {cv_result['accuracy_mean']:.4f} +/- {cv_result['accuracy_std']:.4f}")
    print("CV balanced accuracy scores:", np.round(cv_result["balanced_accuracy_scores"], 4).tolist())
    print(
        f"Mean CV balanced accuracy: {cv_result['balanced_accuracy_mean']:.4f} +/- {cv_result['balanced_accuracy_std']:.4f}"
    )


if __name__ == "__main__":
    main()
