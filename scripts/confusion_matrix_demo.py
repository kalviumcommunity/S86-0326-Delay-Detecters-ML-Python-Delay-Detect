"""Lesson 5.29 demo: creating and interpreting confusion matrices.

Run:
    python scripts/confusion_matrix_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.confusion_matrix_evaluation import (
    compute_confusion_matrix,
    confusion_metrics_from_cells,
    extract_binary_confusion_cells,
    most_confused_class_pairs,
    normalize_confusion_matrix,
    threshold_confusion_table,
)
from src.data_preprocessing import clean_data, load_data, split_data
from src.logistic_classification import build_logistic_pipeline


def _print_binary_matrix(cm: np.ndarray) -> None:
    print("[ [TN, FP],")
    print("  [FN, TP] ]")
    print(cm)


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

    print("\n1) Majority baseline confusion matrix")
    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    cm_base = compute_confusion_matrix(y_test, y_base)
    _print_binary_matrix(cm_base)

    base_cells = extract_binary_confusion_cells(y_test, y_base)
    print("Cells:", base_cells)
    print("Derived metrics:", {k: round(v, 4) for k, v in confusion_metrics_from_cells(
        tp=base_cells["tp"], fp=base_cells["fp"], fn=base_cells["fn"], tn=base_cells["tn"]
    ).items()})

    print("\n2) Logistic model confusion matrix (threshold=0.5)")
    model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm_model = compute_confusion_matrix(y_test, y_pred)
    _print_binary_matrix(cm_model)
    model_cells = extract_binary_confusion_cells(y_test, y_pred)
    model_metrics = confusion_metrics_from_cells(
        tp=model_cells["tp"],
        fp=model_cells["fp"],
        fn=model_cells["fn"],
        tn=model_cells["tn"],
    )
    print("Cells:", model_cells)
    print("Derived metrics:", {k: round(v, 4) for k, v in model_metrics.items()})

    print("\n3) Normalized confusion matrix (row-wise recall view)")
    cm_true_norm = normalize_confusion_matrix(cm_model, normalize="true")
    print(np.round(cm_true_norm, 4))

    print("\n4) Threshold impact on confusion cells")
    threshold_rows = threshold_confusion_table(y_test, y_prob, thresholds=[0.3, 0.5, 0.7])
    for row in threshold_rows:
        print(
            f"Threshold {row['threshold']:.1f} | "
            f"TP:{int(row['tp']):4d} FP:{int(row['fp']):4d} "
            f"FN:{int(row['fn']):4d} TN:{int(row['tn']):4d}"
        )

    print("\n5) Multi-class confusion interpretation demo")
    synthetic_y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    synthetic_y_pred = np.array([0, 0, 1, 0, 1, 2, 1, 1, 2, 1, 2, 0])
    labels = ["Class_A", "Class_B", "Class_C"]

    cm_multi = compute_confusion_matrix(synthetic_y_true, synthetic_y_pred, labels=[0, 1, 2])
    print(cm_multi)
    top_confusions = most_confused_class_pairs(cm_multi, class_labels=labels, top_k=3)
    print("Top confused class pairs:")
    for pair in top_confusions:
        print(
            f"actual={pair['actual_class']}, "
            f"predicted={pair['predicted_class']}, "
            f"count={pair['count']}"
        )

    print("\n6) Validation split threshold selection example")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
        stratify=y_train,
    )
    val_model = build_logistic_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
    )
    val_model.fit(X_train_sub, y_train_sub)
    y_val_prob = val_model.predict_proba(X_val)[:, 1]

    val_table = threshold_confusion_table(y_val, y_val_prob, thresholds=[0.3, 0.5, 0.7])
    for row in val_table:
        print(
            f"VAL threshold {row['threshold']:.1f} -> TP:{int(row['tp'])}, FP:{int(row['fp'])}, "
            f"FN:{int(row['fn'])}, TN:{int(row['tn'])}"
        )


if __name__ == "__main__":
    main()