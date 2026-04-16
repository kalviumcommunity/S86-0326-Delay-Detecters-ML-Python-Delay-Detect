"""Feature-importance analysis helpers for tree-based models."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import Config


def _all_raw_features() -> list[str]:
    return list(Config.NUMERICAL_COLS) + list(Config.CATEGORICAL_COLS)


def map_expanded_to_raw_feature(feature_name: str) -> str:
    """Map expanded one-hot feature names back to their raw feature name."""
    raw_features = _all_raw_features()

    if feature_name in raw_features:
        return feature_name

    # Longest-first match avoids ambiguous prefix collisions.
    for raw in sorted(raw_features, key=len, reverse=True):
        if feature_name.startswith(raw + "_"):
            return raw

    return feature_name


def build_random_forest_classifier_pipeline(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = Config.RANDOM_STATE,
) -> Pipeline:
    """Build a leakage-safe preprocessing plus random-forest classification pipeline."""
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

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_random_forest_pipeline(model_pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Fit the random-forest pipeline and return it."""
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


def get_expanded_feature_names(model_pipeline: Pipeline) -> list[str]:
    """Return model feature names after preprocessing expansion."""
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline is missing a preprocessor step")

    names = []
    for raw_name in preprocessor.get_feature_names_out():
        if "__" in raw_name:
            names.append(raw_name.split("__", 1)[1])
        else:
            names.append(raw_name)
    return names


def get_mdi_importance_table(model_pipeline: Pipeline) -> pd.DataFrame:
    """Return impurity-based feature importance table sorted descending."""
    model = model_pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        raise ValueError("Pipeline must contain a fitted tree-based model with feature_importances_")

    names = get_expanded_feature_names(model_pipeline)
    importances = model.feature_importances_

    std = np.zeros_like(importances)
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        per_tree = np.vstack([est.feature_importances_ for est in model.estimators_])
        std = per_tree.std(axis=0)

    return (
        pd.DataFrame(
            {
                "Feature": names,
                "MDI_Importance": importances,
                "MDI_Std": std,
            }
        )
        .sort_values("MDI_Importance", ascending=False)
        .reset_index(drop=True)
    )


def get_permutation_importance_table(
    model_pipeline: Pipeline,
    X_eval,
    y_eval,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = Config.RANDOM_STATE,
) -> pd.DataFrame:
    """Return permutation-importance table sorted descending."""
    result = permutation_importance(
        model_pipeline,
        X_eval,
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    names = None
    if hasattr(X_eval, "columns"):
        names = list(X_eval.columns)

    if names is None or len(names) != len(result.importances_mean):
        expanded = get_expanded_feature_names(model_pipeline)
        if len(expanded) == len(result.importances_mean):
            names = expanded
        else:
            names = [f"feature_{idx}" for idx in range(len(result.importances_mean))]

    return (
        pd.DataFrame(
            {
                "Feature": names,
                "Permutation_Importance": result.importances_mean,
                "Permutation_Std": result.importances_std,
            }
        )
        .sort_values("Permutation_Importance", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_importance_to_raw_features(
    importance_df: pd.DataFrame,
    importance_col: str,
    std_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate expanded feature importance scores to raw feature names."""
    temp = importance_df.copy()
    temp["Raw_Feature"] = temp["Feature"].map(map_expanded_to_raw_feature)

    grouped = temp.groupby("Raw_Feature", as_index=False)[importance_col].sum()

    if std_col is not None and std_col in temp.columns:
        # Conservative combination for uncertainty across grouped components.
        temp["_std_sq"] = temp[std_col] ** 2
        std_grouped = temp.groupby("Raw_Feature", as_index=False)["_std_sq"].sum()
        std_grouped[std_col] = np.sqrt(std_grouped["_std_sq"])
        std_grouped = std_grouped.drop(columns=["_std_sq"])
        grouped = grouped.merge(std_grouped, on="Raw_Feature", how="left")

    rename_map = {"Raw_Feature": "Feature"}
    grouped = grouped.rename(columns=rename_map)
    return grouped.sort_values(importance_col, ascending=False).reset_index(drop=True)


def compare_importance_tables(mdi_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    """Merge impurity and permutation tables and add rank differences."""
    mdi_ranked = aggregate_importance_to_raw_features(mdi_df, "MDI_Importance", std_col="MDI_Std")
    mdi_ranked["MDI_Rank"] = np.arange(1, len(mdi_ranked) + 1)

    perm_ranked = aggregate_importance_to_raw_features(
        perm_df,
        "Permutation_Importance",
        std_col="Permutation_Std",
    )
    perm_ranked["Permutation_Rank"] = np.arange(1, len(perm_ranked) + 1)

    merged = mdi_ranked.merge(perm_ranked, on="Feature", how="inner")
    merged["Rank_Delta_MDI_minus_Permutation"] = merged["MDI_Rank"] - merged["Permutation_Rank"]

    return merged.sort_values("MDI_Rank").reset_index(drop=True)


def find_high_correlation_pairs(X_numeric: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Return numeric feature pairs with absolute correlation >= threshold."""
    corr = X_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rows = []
    for col in upper.columns:
        strong = upper[col][upper[col] >= threshold]
        for row_name, corr_value in strong.items():
            rows.append(
                {
                    "Feature_A": row_name,
                    "Feature_B": col,
                    "Abs_Correlation": float(corr_value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Feature_A", "Feature_B", "Abs_Correlation"])

    return pd.DataFrame(rows).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)


def evaluate_drop_impact(
    X_train,
    X_test,
    y_train,
    y_test,
    drop_raw_features: list[str],
    random_state: int = Config.RANDOM_STATE,
) -> Dict[str, float | list[str]]:
    """Retrain with selected raw features removed and report test-accuracy impact."""
    drop_set = set(drop_raw_features)

    kept_numeric = [c for c in Config.NUMERICAL_COLS if c not in drop_set]
    kept_categorical = [c for c in Config.CATEGORICAL_COLS if c not in drop_set]

    baseline = build_random_forest_classifier_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        random_state=random_state,
    )
    baseline.fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)

    dropped = build_random_forest_classifier_pipeline(
        categorical_cols=kept_categorical,
        numerical_cols=kept_numeric,
        random_state=random_state,
    )
    dropped.fit(X_train[kept_numeric + kept_categorical], y_train)
    dropped_acc = dropped.score(X_test[kept_numeric + kept_categorical], y_test)

    return {
        "baseline_accuracy": float(baseline_acc),
        "dropped_accuracy": float(dropped_acc),
        "accuracy_delta": float(dropped_acc - baseline_acc),
        "dropped_features": sorted(list(drop_set)),
    }
