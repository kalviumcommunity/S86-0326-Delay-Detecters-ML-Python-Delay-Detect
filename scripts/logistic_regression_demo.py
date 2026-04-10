"""Lesson 5.25 demo: majority baseline vs Logistic Regression classification.

Run:
    python scripts/logistic_regression_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.logistic_classification import (
    build_logistic_pipeline,
    compare_model_vs_baseline,
    cross_validate_auc_f1,
    evaluate_binary_classifier,
    extract_logistic_coefficients,
    train_majority_baseline,
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

    print("\n1) Majority baseline")
    baseline = train_majority_baseline(X_train, y_train, random_state=Config.RANDOM_STATE)
    baseline_pred = baseline.predict(X_test)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_binary_classifier(y_test, baseline_pred, baseline_prob)

    print("\n2) Logistic Regression pipeline")
    model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    model_pred = model.predict(X_test)
    model_prob = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate_binary_classifier(y_test, model_pred, model_prob)

    print("\n3) Metrics")
    print("Baseline:")
    print({k: round(v, 4) for k, v in baseline_metrics.items()})
    print("Logistic Regression:")
    print({k: round(v, 4) for k, v in model_metrics.items()})

    improvement = compare_model_vs_baseline(model_metrics, baseline_metrics)
    print("Improvement:")
    print({k: round(v, 4) for k, v in improvement.items()})

    print("\nClassification report (Logistic Regression):")
    print(classification_report(y_test, model_pred, zero_division=0))

    print("\n4) Cross-validation (AUC/F1)")
    cv_result = cross_validate_auc_f1(model, X_train, y_train, cv=5)
    print("CV AUC scores:", np.round(cv_result["auc_scores"], 4).tolist())
    print(f"Mean CV AUC: {cv_result['auc_mean']:.4f} +/- {cv_result['auc_std']:.4f}")
    print("CV F1 scores:", np.round(cv_result["f1_scores"], 4).tolist())
    print(f"Mean CV F1: {cv_result['f1_mean']:.4f} +/- {cv_result['f1_std']:.4f}")

    print("\n5) Top coefficients (odds ratio view)")
    coef_df = extract_logistic_coefficients(model)
    print(f"Intercept: {model.named_steps['model'].intercept_[0]:.4f}")
    print(coef_df[["feature", "coefficient", "odds_ratio"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
