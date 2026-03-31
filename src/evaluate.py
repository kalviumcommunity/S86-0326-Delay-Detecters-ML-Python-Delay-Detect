"""
Model evaluation module.

This module handles model evaluation on held-out test data. It computes
metrics and returns them as structured data (dictionaries), not printed output.
This allows metrics to be logged, compared, and aggregated programmatically.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Union, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import Config


def evaluate_model(
    model: Union[RandomForestClassifier, LogisticRegression],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[str] = None
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    This function computes evaluation metrics and returns them as a dictionary.
    It does not print results (printing happens only in orchestration/reporting).
    This allows metrics to be captured, logged, and programmed against.
    
    Parameters:
        model: Fitted model object
        X_test: Test feature array
        y_test: Test target array
        metrics: List of metric names to compute. If None, uses Config defaults.
    
    Returns:
        Dictionary mapping metric names to values.
        
    Raises:
        ValueError: If y_test contains values not seen during training
        ValueError: If metrics contains unsupported metric names
    """
    if metrics is None:
        metrics = Config.EVAL_METRICS
    
    # Validate input
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test must have same length. Got {len(X_test)} and {len(y_test)}"
        )
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Try to get probabilities for AUC (binary classification only)
    if len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    else:
        y_proba = None
    
    # Compute metrics
    results = {}
    
    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = accuracy_score(y_test, y_pred)
        
        elif metric == "precision":
            results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        
        elif metric == "recall":
            results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        
        elif metric == "f1":
            results["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        elif metric == "roc_auc":
            if y_proba is not None:
                results["roc_auc"] = roc_auc_score(y_test, y_proba)
            else:
                results["roc_auc"] = None  # Not applicable for multiclass without probabilities
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    return results


def compute_confusion_matrix(
    model: Union[RandomForestClassifier, LogisticRegression],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix for test set predictions.
    
    Parameters:
        model: Fitted model object
        X_test: Test feature array
        y_test: Test target array
    
    Returns:
        Confusion matrix as numpy array
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def get_classification_report(
    model: Union[RandomForestClassifier, LogisticRegression],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> str:
    """
    Generate detailed classification report for test set.
    
    Parameters:
        model: Fitted model object
        X_test: Test feature array
        y_test: Test target array
    
    Returns:
        Classification report as formatted string
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report


def compare_metrics(
    metrics_dict: Dict[str, float],
    baseline_dict: Dict[str, float] = None
) -> Dict[str, dict]:
    """
    Compare current metrics against baseline.
    
    Useful for tracking model improvements over iterations.
    
    Parameters:
        metrics_dict: Current model metrics
        baseline_dict: Previous/baseline metrics to compare against
    
    Returns:
        Dictionary with metric names as keys and dicts containing current,
        baseline, and improvement values
    """
    if baseline_dict is None:
        raise ValueError("baseline_dict must be provided for comparison")
    
    comparison = {}
    for metric_name, current_value in metrics_dict.items():
        if metric_name in baseline_dict:
            baseline_value = baseline_dict[metric_name]
            improvement = current_value - baseline_value
            improvement_pct = (improvement / baseline_value * 100) if baseline_value != 0 else 0
            
            comparison[metric_name] = {
                "current": current_value,
                "baseline": baseline_value,
                "improvement": improvement,
                "improvement_pct": improvement_pct
            }
        else:
            comparison[metric_name] = {
                "current": current_value,
                "baseline": None,
                "improvement": None,
                "improvement_pct": None
            }
    
    return comparison
