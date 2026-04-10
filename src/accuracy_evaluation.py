"""Accuracy-focused utilities for classification model evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_accuracy_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard and balanced accuracy metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def confusion_matrix_breakdown(y_true, y_pred) -> Dict[str, int]:
    """Return confusion-matrix cell counts for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("confusion_matrix_breakdown supports binary classification only")

    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def majority_class_baseline_accuracy(y) -> float:
    """Compute expected accuracy from always predicting the majority class."""
    values, counts = np.unique(y, return_counts=True)
    _ = values
    majority_count = int(np.max(counts))
    return majority_count / len(y)


def compare_with_baseline_accuracy(model_acc: float, baseline_acc: float) -> Dict[str, float]:
    """Compute absolute and relative gain over baseline accuracy."""
    gain = model_acc - baseline_acc
    if baseline_acc == 0:
        rel_gain_pct = 0.0
    else:
        rel_gain_pct = (gain / baseline_acc) * 100.0

    return {
        "accuracy_gain": gain,
        "relative_gain_pct": rel_gain_pct,
    }


def cross_validate_accuracy(
    model,
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run stratified CV and return accuracy + balanced accuracy summaries."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    acc_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    bal_scores = cross_val_score(model, X, y, cv=skf, scoring="balanced_accuracy")

    return {
        "accuracy_scores": acc_scores,
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "balanced_accuracy_scores": bal_scores,
        "balanced_accuracy_mean": float(np.mean(bal_scores)),
        "balanced_accuracy_std": float(np.std(bal_scores)),
    }
