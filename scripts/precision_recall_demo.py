"""Lesson 5.27 demo: evaluating classification with precision and recall.

Run:
    python scripts/precision_recall_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.logistic_classification import build_logistic_pipeline
from src.precision_recall_evaluation import (
    best_threshold_for_min_recall,
    cross_validate_precision_recall,
    evaluate_fbeta,
    evaluate_precision_recall,
    precision_recall_curve_data,
    threshold_predictions,
)


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

    print("\n1) Majority baseline precision/recall")
    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_precision_recall(y_test, baseline_pred)
    print({k: round(v, 4) for k, v in baseline_metrics.items()})

    print("\n2) Logistic Regression at default threshold 0.5")
    model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = threshold_predictions(y_prob, threshold=0.5)
    default_metrics = evaluate_precision_recall(y_test, y_pred)
    f2_default = evaluate_fbeta(y_test, y_pred, beta=2.0)

    print({k: round(v, 4) for k, v in default_metrics.items()})
    print(f"F2-score (recall-weighted): {f2_default:.4f}")

    print("\n3) Threshold tuning for minimum recall")
    target_recall = 0.80
    best = best_threshold_for_min_recall(y_test, y_prob, min_recall=target_recall)
    print("Best threshold result:", {k: (None if v is None else round(v, 4)) for k, v in best.items()})

    if best["threshold"] is not None:
        tuned_pred = threshold_predictions(y_prob, threshold=best["threshold"])
        tuned_metrics = evaluate_precision_recall(y_test, tuned_pred)
        f2_tuned = evaluate_fbeta(y_test, tuned_pred, beta=2.0)

        print("Tuned threshold metrics:")
        print({k: round(v, 4) for k, v in tuned_metrics.items()})
        print(f"Tuned F2-score: {f2_tuned:.4f}")

    print("\nClassification report (default threshold):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n4) PR curve data snapshot")
    pr_data = precision_recall_curve_data(y_test, y_prob)
    print("Curve points:", len(pr_data["precisions"]))
    print("First 5 precision values:", np.round(pr_data["precisions"][:5], 4).tolist())
    print("First 5 recall values:", np.round(pr_data["recalls"][:5], 4).tolist())

    print("\n5) Cross-validation precision/recall stability")
    cv_result = cross_validate_precision_recall(model, X_train, y_train, cv=5, random_state=Config.RANDOM_STATE)
    print("CV precision:", np.round(cv_result["precision_scores"], 4).tolist())
    print(f"Mean precision: {cv_result['precision_mean']:.4f} +/- {cv_result['precision_std']:.4f}")
    print("CV recall:", np.round(cv_result["recall_scores"], 4).tolist())
    print(f"Mean recall: {cv_result['recall_mean']:.4f} +/- {cv_result['recall_std']:.4f}")
    print("CV F1:", np.round(cv_result["f1_scores"], 4).tolist())
    print(f"Mean F1: {cv_result['f1_mean']:.4f} +/- {cv_result['f1_std']:.4f}")


if __name__ == "__main__":
    main()
