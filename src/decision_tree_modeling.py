"""Decision tree training and evaluation utilities for classification and regression."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import Config


def build_decision_tree_classifier_pipeline(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    criterion: str = "gini",
    max_depth: int | None = 4,
    min_samples_leaf: int = 5,
    min_samples_split: int = 2,
    random_state: int = Config.RANDOM_STATE,
) -> Pipeline:
    """Build a leakage-safe preprocessing plus decision tree classification pipeline."""
    if categorical_cols is None:
        categorical_cols = Config.CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = Config.NUMERICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_decision_tree_regressor(
    criterion: str = "squared_error",
    max_depth: int | None = 4,
    min_samples_leaf: int = 5,
    min_samples_split: int = 2,
    random_state: int = Config.RANDOM_STATE,
) -> DecisionTreeRegressor:
    """Build a decision tree regressor for numeric feature matrices."""
    return DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )


def evaluate_decision_tree_classification(y_true, y_pred) -> Dict[str, float]:
    """Compute core classification metrics for decision tree predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_decision_tree_regression(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics for decision tree predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_classification_accuracy(model_accuracy: float, baseline_accuracy: float) -> Dict[str, float]:
    """Return absolute and relative accuracy gain versus a baseline."""
    gain = model_accuracy - baseline_accuracy
    if baseline_accuracy == 0:
        relative_gain_pct = 0.0 if gain == 0 else float("inf")
    else:
        relative_gain_pct = (gain / baseline_accuracy) * 100.0

    return {
        "accuracy_gain": gain,
        "relative_gain_pct": relative_gain_pct,
    }


def tune_decision_tree_classifier_depth(
    model_pipeline: Pipeline,
    X_train,
    y_train,
    depth_values: Iterable[int],
    cv: int = 5,
    random_state: int = Config.RANDOM_STATE,
) -> Dict[str, object]:
    """Evaluate candidate depths via stratified cross-validation and return the best depth."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    depth_list = list(depth_values)

    cv_scores = []
    train_scores = []

    for depth in depth_list:
        tuned = clone(model_pipeline).set_params(model__max_depth=depth)
        cv_result = cross_validate(
            tuned,
            X_train,
            y_train,
            cv=skf,
            scoring="accuracy",
            return_train_score=True,
        )

        cv_scores.append(float(np.mean(cv_result["test_score"])))
        train_scores.append(float(np.mean(cv_result["train_score"])))

    best_index = int(np.argmax(cv_scores))

    return {
        "depth_values": depth_list,
        "mean_cv_scores": cv_scores,
        "mean_train_scores": train_scores,
        "best_depth": int(depth_list[best_index]),
        "best_cv_score": float(cv_scores[best_index]),
    }


def cross_validate_decision_tree_classifier(
    model_pipeline: Pipeline,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = Config.RANDOM_STATE,
) -> Dict[str, object]:
    """Cross-validate a decision tree classifier and report stability summaries."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    accuracy_scores = cross_val_score(model_pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    f1_scores = cross_val_score(model_pipeline, X_train, y_train, cv=skf, scoring="f1")

    return {
        "accuracy_scores": accuracy_scores,
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores)),
        "f1_scores": f1_scores,
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
    }


def cross_validate_decision_tree_regression(
    model: DecisionTreeRegressor,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = Config.RANDOM_STATE,
) -> Dict[str, object]:
    """Cross-validate a decision tree regressor and report RMSE/R2 summaries."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    neg_rmse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    rmse_scores = -neg_rmse_scores

    return {
        "rmse_scores": rmse_scores,
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_scores": r2_scores,
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
    }


def get_tree_feature_names(pipeline: Pipeline) -> list[str]:
    """Return human-readable feature names from a fitted tree classification pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline is missing a preprocessor step")

    feature_names = []
    for name in preprocessor.get_feature_names_out():
        if "__" in name:
            feature_names.append(name.split("__", 1)[1])
        else:
            feature_names.append(name)
    return feature_names


def build_feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame:
    """Build a sorted feature-importance table for a fitted decision tree pipeline."""
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        raise ValueError("Pipeline must contain a fitted tree model with feature_importances_")

    return (
        pd.DataFrame(
            {
                "Feature": get_tree_feature_names(pipeline),
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )