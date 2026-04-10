"""Confusion-matrix focused evaluation helpers for classification models."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Compute confusion matrix with optional explicit label ordering."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def extract_binary_confusion_cells(y_true, y_pred) -> Dict[str, int]:
    """Extract TN, FP, FN, TP from a binary confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def confusion_metrics_from_cells(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Derive core binary metrics directly from confusion-matrix cells."""
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }


def normalize_confusion_matrix(cm: np.ndarray, normalize: str = "true") -> np.ndarray:
    """Normalize confusion matrix by rows, columns, or globally."""
    matrix = np.asarray(cm, dtype=float)

    if normalize == "true":
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    if normalize == "pred":
        col_sums = matrix.sum(axis=0, keepdims=True)
        return np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums != 0)

    if normalize == "all":
        total = matrix.sum()
        return matrix / total if total else np.zeros_like(matrix)

    if normalize is None:
        return matrix

    raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


def threshold_confusion_table(y_true, y_prob, thresholds: List[float]) -> List[Dict[str, float]]:
    """Compute confusion-cell counts across multiple probability thresholds."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        cells = extract_binary_confusion_cells(y_true_arr, y_pred)
        rows.append(
            {
                "threshold": float(threshold),
                "tp": float(cells["tp"]),
                "fp": float(cells["fp"]),
                "fn": float(cells["fn"]),
                "tn": float(cells["tn"]),
            }
        )

    return rows


def most_confused_class_pairs(cm: np.ndarray, class_labels: List[str], top_k: int = 3) -> List[Dict[str, object]]:
    """Return top off-diagonal confusion pairs for multi-class inspection."""
    matrix = np.asarray(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    if len(class_labels) != matrix.shape[0]:
        raise ValueError("class_labels length must match confusion-matrix size.")

    pairs: List[Dict[str, object]] = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            pairs.append(
                {
                    "actual_class": class_labels[i],
                    "predicted_class": class_labels[j],
                    "count": int(matrix[i, j]),
                }
            )

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]