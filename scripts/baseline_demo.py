"""Lesson 5.21 baseline demo.

Compares a majority-class baseline, a simple heuristic baseline, and a trained
model on the same test split.

Run:
    python scripts/baseline_demo.py
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add project root for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import (
    compare_to_baseline,
    evaluate_classification_baseline,
    train_classification_baseline,
)
from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import prepare_features


def heuristic_predict(X_test, distance_threshold: float) -> np.ndarray:
    """Simple business-rule heuristic using only distance."""
    if "distance_km" not in X_test.columns:
        return np.zeros(len(X_test), dtype=int)
    return (X_test["distance_km"] >= distance_threshold).astype(int).to_numpy()


def main() -> None:
    print("Loading and splitting data...")
    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    print("\n1) Majority-class baseline")
    baseline = train_classification_baseline(X_train, y_train, strategy="most_frequent")
    baseline_metrics = evaluate_classification_baseline(baseline, X_test, y_test)
    print({k: (None if v is None else round(v, 4)) for k, v in baseline_metrics.items()})

    print("\n2) Heuristic baseline (distance threshold from training median)")
    threshold = float(X_train["distance_km"].median()) if "distance_km" in X_train.columns else 0.0
    heuristic_pred = heuristic_predict(X_test, threshold)
    heuristic_metrics = {
        "accuracy": accuracy_score(y_test, heuristic_pred),
        "f1": f1_score(y_test, heuristic_pred, average="weighted", zero_division=0),
    }
    print({k: round(v, 4) for k, v in heuristic_metrics.items()})

    print("\n3) Trained model (LogisticRegression)")
    X_train_prepared, pipeline = prepare_features(X_train, fit_pipeline=True)
    X_test_prepared, _ = prepare_features(
        X_test,
        fit_pipeline=False,
        preprocessing_pipeline=pipeline,
    )

    model = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)
    model.fit(X_train_prepared, y_train)
    model_pred = model.predict(X_test_prepared)
    model_metrics = evaluate_model(model, X_test_prepared, y_test)
    print({k: (None if v is None else round(v, 4)) for k, v in model_metrics.items()})

    improvement = compare_to_baseline(model_metrics, baseline_metrics)
    print("\nImprovement vs majority baseline:")
    print({k: (None if v is None else round(v, 4)) for k, v in improvement.items()})

    print("\nBaseline classification report:")
    print(classification_report(y_test, baseline.predict(X_test), zero_division=0))
    print("Model classification report:")
    print(classification_report(y_test, model_pred, zero_division=0))


if __name__ == "__main__":
    main()
