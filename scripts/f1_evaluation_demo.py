"""Lesson 5.28 demo: evaluating classification models using F1-score.

Run:
    python scripts/f1_evaluation_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.f1_evaluation import (
    compare_f1_vs_baseline,
    cross_validate_f1,
    evaluate_binary_f1_metrics,
    evaluate_f1_averages,
    find_best_threshold_for_f1,
    threshold_predictions,
)
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

    print("\n1) Majority baseline")
    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_binary_f1_metrics(y_test, baseline_pred)
    print({k: round(v, 4) for k, v in baseline_metrics.items()})

    print("\n2) Logistic Regression (default threshold=0.5)")
    model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_default = threshold_predictions(y_prob_test, threshold=0.5)
    model_metrics = evaluate_binary_f1_metrics(y_test, y_pred_default)
    f1_avgs = evaluate_f1_averages(y_test, y_pred_default)

    print({k: round(v, 4) for k, v in model_metrics.items()})
    print({k: round(v, 4) for k, v in f1_avgs.items()})

    gain = compare_f1_vs_baseline(
        model_f1=model_metrics["f1"],
        baseline_f1=baseline_metrics["f1"],
    )
    print("F1 gain over baseline:")
    printable_gain = {
        k: ("inf" if np.isinf(v) else round(v, 4)) for k, v in gain.items()
    }
    print(printable_gain)

    print("\n3) Threshold tuning on validation set (not test)")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
        stratify=y_train,
    )

    tuned_model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    tuned_model.fit(X_train_sub, y_train_sub)

    y_prob_val = tuned_model.predict_proba(X_val)[:, 1]
    best = find_best_threshold_for_f1(y_val, y_prob_val)
    print({k: round(v, 4) for k, v in best.items()})

    y_prob_test_tuned = tuned_model.predict_proba(X_test)[:, 1]
    y_pred_tuned = threshold_predictions(y_prob_test_tuned, threshold=best["best_threshold"])
    tuned_metrics = evaluate_binary_f1_metrics(y_test, y_pred_tuned)
    print("Tuned-threshold test metrics:")
    print({k: round(v, 4) for k, v in tuned_metrics.items()})

    print("\nClassification report (default threshold):")
    print(classification_report(y_test, y_pred_default, zero_division=0))

    print("\n4) Cross-validation F1 stability")
    cv_result = cross_validate_f1(model, X_train, y_train, cv=5, random_state=Config.RANDOM_STATE)
    print("CV binary F1:", np.round(cv_result["f1_binary_scores"], 4).tolist())
    print(f"Mean binary F1: {cv_result['f1_binary_mean']:.4f} +/- {cv_result['f1_binary_std']:.4f}")
    print("CV macro F1:", np.round(cv_result["f1_macro_scores"], 4).tolist())
    print(f"Mean macro F1: {cv_result['f1_macro_mean']:.4f} +/- {cv_result['f1_macro_std']:.4f}")
    print("CV weighted F1:", np.round(cv_result["f1_weighted_scores"], 4).tolist())
    print(f"Mean weighted F1: {cv_result['f1_weighted_mean']:.4f} +/- {cv_result['f1_weighted_std']:.4f}")


if __name__ == "__main__":
    main()
