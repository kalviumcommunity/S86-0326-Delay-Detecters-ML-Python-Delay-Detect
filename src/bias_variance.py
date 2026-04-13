"""Bias-variance diagnostics for train/test and learning-curve behavior."""

from __future__ import annotations

from typing import Dict

import numpy as np


def diagnose_train_test_behavior(
    train_score: float,
    test_score: float,
    good_score_threshold: float = 0.8,
    large_gap_threshold: float = 0.1,
) -> Dict[str, float | str]:
    """Diagnose likely bias/variance state from train/test scores.

    Assumes a higher score is better (for example, accuracy/F1/R2).
    """
    gap = train_score - test_score
    abs_gap = abs(gap)

    if train_score < good_score_threshold and test_score < good_score_threshold and abs_gap <= large_gap_threshold:
        diagnosis = "high_bias"
        recommendation = "Increase model flexibility, improve features, or reduce regularization."
    elif train_score >= good_score_threshold and gap >= large_gap_threshold:
        diagnosis = "high_variance"
        recommendation = "Reduce complexity, regularize, and/or add more training data."
    elif train_score >= good_score_threshold and test_score >= good_score_threshold and abs_gap < large_gap_threshold:
        diagnosis = "good_fit"
        recommendation = "Model is balanced; continue with careful tuning and monitoring."
    else:
        diagnosis = "mixed_or_unclear"
        recommendation = "Check data quality, leakage risk, and run learning-curve diagnostics."

    return {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "gap": float(gap),
        "abs_gap": float(abs_gap),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def summarize_cv_stability(cv_scores: np.ndarray, high_variability_threshold: float = 0.03) -> Dict[str, float | str]:
    """Return CV mean/std plus a simple stability label."""
    mean_score = float(np.mean(cv_scores))
    std_score = float(np.std(cv_scores))

    stability = "stable" if std_score <= high_variability_threshold else "unstable"

    return {
        "mean": mean_score,
        "std": std_score,
        "stability": stability,
    }


def interpret_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    good_score_threshold: float = 0.8,
    large_gap_threshold: float = 0.1,
) -> Dict[str, float | str]:
    """Interpret final-point train/validation learning-curve behavior.

    Expects arrays with shape (n_train_sizes, n_folds).
    """
    train_mean_curve = train_scores.mean(axis=1)
    val_mean_curve = val_scores.mean(axis=1)

    final_train = float(train_mean_curve[-1])
    final_val = float(val_mean_curve[-1])
    final_gap = final_train - final_val

    diagnosis = diagnose_train_test_behavior(
        train_score=final_train,
        test_score=final_val,
        good_score_threshold=good_score_threshold,
        large_gap_threshold=large_gap_threshold,
    )

    return {
        "final_train": final_train,
        "final_val": final_val,
        "final_gap": float(final_gap),
        "diagnosis": diagnosis["diagnosis"],
        "recommendation": diagnosis["recommendation"],
    }


def decompose_expected_error(bias_sq: float, variance: float, noise: float) -> float:
    """Compute total expected prediction error from decomposition terms."""
    return float(bias_sq + variance + noise)
