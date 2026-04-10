"""Logistic Regression training and evaluation utilities for binary classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config


def build_logistic_pipeline(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    C: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-safe preprocessing + logistic regression pipeline."""
    if categorical_cols is None:
        categorical_cols = Config.CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = Config.NUMERICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


def train_majority_baseline(X_train, y_train, random_state: int = 42) -> DummyClassifier:
    """Fit a majority-class baseline classifier."""
    baseline = DummyClassifier(strategy="most_frequent", random_state=random_state)
    baseline.fit(X_train, y_train)
    return baseline


def evaluate_binary_classifier(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute core binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def compare_model_vs_baseline(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute metric deltas versus baseline."""
    return {
        "accuracy_gain": model_metrics["accuracy"] - baseline_metrics["accuracy"],
        "f1_gain": model_metrics["f1"] - baseline_metrics["f1"],
        "roc_auc_gain": model_metrics["roc_auc"] - baseline_metrics["roc_auc"],
    }


def cross_validate_auc_f1(model_pipeline: Pipeline, X_train, y_train, cv: int = 5) -> Dict[str, object]:
    """Cross-validate ROC-AUC and F1 with mean/std summaries."""
    auc_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    f1_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="f1")

    return {
        "auc_scores": auc_scores,
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "f1_scores": f1_scores,
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
    }


def extract_logistic_coefficients(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Extract fitted logistic coefficients and odds ratios from pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coef = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "odds_ratio": np.exp(coef),
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df
