"""F1-score focused evaluation helpers for classification models."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_binary_f1_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute precision, recall, and F1 for binary predictions."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_f1_averages(y_true, y_pred) -> Dict[str, float]:
    """Compute micro, macro, and weighted F1 for general classification."""
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_fbeta(y_true, y_pred, beta: float = 2.0) -> float:
    """Compute F-beta score for custom precision/recall weighting."""
    return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)


def threshold_predictions(y_prob, threshold: float = 0.5):
    """Convert positive-class probabilities to binary labels at threshold."""
    return (np.asarray(y_prob) >= threshold).astype(int)


def find_best_threshold_for_f1(y_true, y_prob, thresholds=None) -> Dict[str, float]:
    """Find threshold that maximizes binary F1 on provided validation labels."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = threshold_predictions(y_prob, threshold=threshold)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return {
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
    }


def compare_f1_vs_baseline(model_f1: float, baseline_f1: float) -> Dict[str, float]:
    """Compute absolute and relative F1 improvements over baseline."""
    gain = model_f1 - baseline_f1
    if baseline_f1 == 0:
        rel_gain_pct = 0.0 if gain == 0 else float("inf")
    else:
        rel_gain_pct = (gain / baseline_f1) * 100.0

    return {
        "f1_gain": gain,
        "relative_gain_pct": rel_gain_pct,
    }


def cross_validate_f1(model, X, y, cv: int = 5, random_state: int = 42) -> Dict[str, object]:
    """Run stratified CV and return binary/macro/weighted F1 summaries."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    f1_binary = cross_val_score(model, X, y, cv=skf, scoring="f1")
    f1_macro = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
    f1_weighted = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")

    return {
        "f1_binary_scores": f1_binary,
        "f1_binary_mean": float(np.mean(f1_binary)),
        "f1_binary_std": float(np.std(f1_binary)),
        "f1_macro_scores": f1_macro,
        "f1_macro_mean": float(np.mean(f1_macro)),
        "f1_macro_std": float(np.std(f1_macro)),
        "f1_weighted_scores": f1_weighted,
        "f1_weighted_mean": float(np.mean(f1_weighted)),
        "f1_weighted_std": float(np.std(f1_weighted)),
    }
