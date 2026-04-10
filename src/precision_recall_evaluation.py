"""Precision/Recall-focused evaluation helpers for binary classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_precision_recall(y_true, y_pred) -> Dict[str, float]:
    """Compute precision, recall, and F1 for binary predictions."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_fbeta(y_true, y_pred, beta: float = 2.0) -> float:
    """Compute F-beta score to weight recall/precision asymmetrically."""
    return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)


def threshold_predictions(y_prob, threshold: float = 0.5):
    """Convert probabilities to class labels at a custom threshold."""
    return (np.asarray(y_prob) >= threshold).astype(int)


def precision_recall_curve_data(y_true, y_prob) -> Dict[str, np.ndarray]:
    """Return precision-recall curve arrays."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
    }


def best_threshold_for_min_recall(y_true, y_prob, min_recall: float = 0.8) -> Dict[str, float | None]:
    """Pick threshold that maximizes precision under recall constraint."""
    curve = precision_recall_curve_data(y_true, y_prob)
    precisions = curve["precisions"]
    recalls = curve["recalls"]
    thresholds = curve["thresholds"]

    # thresholds has length n-1 relative to precision/recall arrays.
    valid_idx = np.where(recalls[:-1] >= min_recall)[0]
    if len(valid_idx) == 0:
        return {
            "threshold": None,
            "precision": None,
            "recall": None,
        }

    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
    }


def cross_validate_precision_recall(
    model,
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run stratified CV and return precision/recall/F1 summaries."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    precision_scores = cross_val_score(model, X, y, cv=skf, scoring="precision")
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring="recall")
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1")

    return {
        "precision_scores": precision_scores,
        "precision_mean": float(np.mean(precision_scores)),
        "precision_std": float(np.std(precision_scores)),
        "recall_scores": recall_scores,
        "recall_mean": float(np.mean(recall_scores)),
        "recall_std": float(np.std(recall_scores)),
        "f1_scores": f1_scores,
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
    }
