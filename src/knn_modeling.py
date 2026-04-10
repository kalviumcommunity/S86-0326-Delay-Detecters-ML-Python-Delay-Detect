"""KNN training and evaluation utilities for classification and regression."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config


def build_knn_classifier_pipeline(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
) -> Pipeline:
    """Build a leakage-safe preprocessing + KNN classifier pipeline."""
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

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_knn_regressor_pipeline(
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    scale_features: bool = True,
) -> Pipeline:
    """Build KNN regressor pipeline for numeric feature matrices."""
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    steps.append(
        (
            "model",
            KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p,
            ),
        )
    )
    return Pipeline(steps)


def evaluate_knn_classification(y_true, y_pred) -> Dict[str, float]:
    """Compute core binary classification metrics for KNN predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_knn_regression(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics for KNN predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_classification_accuracy(model_accuracy: float, baseline_accuracy: float) -> Dict[str, float]:
    """Return absolute and relative accuracy gain versus baseline."""
    gain = model_accuracy - baseline_accuracy
    if baseline_accuracy == 0:
        relative_gain_pct = 0.0 if gain == 0 else float("inf")
    else:
        relative_gain_pct = (gain / baseline_accuracy) * 100.0

    return {
        "accuracy_gain": gain,
        "relative_gain_pct": relative_gain_pct,
    }


def tune_knn_classifier_k(
    model_pipeline: Pipeline,
    X_train,
    y_train,
    k_values: Iterable[int],
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Evaluate candidate K values via stratified CV and return best K."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    k_list = list(k_values)

    cv_scores = []
    train_scores = []

    for k in k_list:
        tuned = model_pipeline.set_params(model__n_neighbors=k)
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

    best_idx = int(np.argmax(cv_scores))

    return {
        "k_values": k_list,
        "mean_cv_scores": cv_scores,
        "mean_train_scores": train_scores,
        "best_k": int(k_list[best_idx]),
        "best_cv_score": float(cv_scores[best_idx]),
    }


def cross_validate_knn_classifier(
    model_pipeline: Pipeline,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Cross-validate KNN classifier and report stability summaries."""
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


def cross_validate_knn_regression(
    model_pipeline: Pipeline,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Cross-validate KNN regressor and report RMSE/R2 summaries."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    neg_rmse_scores = cross_val_score(model_pipeline, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(model_pipeline, X_train, y_train, cv=kf, scoring="r2")
    rmse_scores = -neg_rmse_scores

    return {
        "rmse_scores": rmse_scores,
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_scores": r2_scores,
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
    }