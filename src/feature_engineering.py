"""
Feature engineering module.

This module handles feature transformation including encoding categorical
variables, scaling numerical features, and constructing preprocessing pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List

from src.config import Config


def encode_categorical_features(
    X: pd.DataFrame,
    categorical_cols: List[str],
    method: str = "onehot"
) -> pd.DataFrame:
    """
    Encode categorical features using specified method.
    
    This function transforms categorical columns to numerical representations.
    Currently supports one-hot encoding. This function applies fitted encoder
    to new data (not fitting a new one).
    
    Parameters:
        X: Feature DataFrame
        categorical_cols: List of categorical column names
        method: Encoding method ("onehot" is currently supported)
    
    Returns:
        DataFrame with encoded categorical features
        
    Raises:
        ValueError: If method is not supported or categorical_cols contain invalid columns
    """
    X = X.copy()
    
    # Validate columns exist
    missing_cols = set(categorical_cols) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Categorical columns not found in DataFrame: {missing_cols}")
    
    if method == "onehot":
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        raise ValueError(f"Unsupported encoding method: {method}")
    
    return X


def scale_numerical_features(
    X: pd.DataFrame,
    numerical_cols: List[str],
    scaler=None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features to zero mean and unit variance.
    
    This function applies standardization to numerical columns. When fit=True,
    it learns scaling parameters from the data (used with training data).
    When fit=False, it applies pre-fit parameters (used during inference).
    
    Parameters:
        X: Feature DataFrame
        numerical_cols: List of numerical column names
        scaler: Pre-fit StandardScaler instance. If None, creates new scaler.
        fit: Whether to fit the scaler (True for training, False for inference)
    
    Returns:
        Tuple of (scaled_DataFrame, fitted_scaler)
        
    Raises:
        ValueError: If numerical_cols contain invalid columns
    """
    X = X.copy()
    
    # Validate columns exist
    missing_cols = set(numerical_cols) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Numerical columns not found in DataFrame: {missing_cols}")
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X, scaler


def create_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    This function engineers new features that may improve model performance.
    Examples include interaction terms, polynomial features, or domain-specific
    transformations.
    
    Parameters:
        X: Feature DataFrame
    
    Returns:
        DataFrame with additional derived features
    """
    X = X.copy()
    
    # Example derived features (customize based on your domain knowledge)
    # These are illustrative - modify based on actual data
    
    # If distance_km and items_count exist, create interaction
    if "distance_km" in X.columns and "items_count" in X.columns:
        X["item_distance_interaction"] = X["distance_km"] * X["items_count"]
    
    # If order_value exists, create value bins
    if "order_value" in X.columns:
        X["high_value_order"] = (X["order_value"] > X["order_value"].median()).astype(int)
    
    return X


def build_preprocessing_pipeline(
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None
) -> Pipeline:
    """
    Construct sklearn ColumnTransformer for data preprocessing.
    
    This function builds a reusable pipeline that handles both numerical
    and categorical feature transformations. This is essential for ensuring
    the same transformations are applied during training and inference.
    
    Parameters:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
    
    Returns:
        sklearn Pipeline object for data transformation
    """
    # Use config defaults if not specified
    if categorical_cols is None:
        categorical_cols = Config.CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = Config.NUMERICAL_COLS
    
    # Define transformers
    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough"
    )
    
    return preprocessor


def apply_preprocessing_pipeline(
    X: pd.DataFrame,
    pipeline: Pipeline,
    fit: bool = False
) -> Tuple[np.ndarray, Pipeline]:
    """
    Apply preprocessing pipeline to features.
    
    This function is a wrapper that makes explicit whether the pipeline
    is being fit (training) or applied (inference). This prevents accidental
    data leakage.
    
    Parameters:
        X: Feature DataFrame
        pipeline: sklearn Pipeline object
        fit: Whether to fit the pipeline (True for training, False for inference)
    
    Returns:
        Tuple of (transformed_array, pipeline)
        
    Raises:
        ValueError: If pipeline is not fitted when fit=False
    """
    if fit:
        X_transformed = pipeline.fit_transform(X)
    else:
        X_transformed = pipeline.transform(X)
    
    return X_transformed, pipeline


def prepare_features(
    X: pd.DataFrame,
    y: pd.Series = None,
    fit_pipeline: bool = True,
    preprocessing_pipeline: Pipeline = None
) -> Tuple[np.ndarray, Pipeline]:
    """
    Complete feature preparation pipeline orchestrator.
    
    This function combines feature engineering and preprocessing in order,
    handling both fitting (during training) and application (during inference).
    
    Parameters:
        X: Feature DataFrame
        y: Target Series (optional, used for potential target encoding)
        fit_pipeline: Whether to fit transformations (True for training data)
        preprocessing_pipeline: Pre-fit pipeline (must be provided if fit_pipeline=False)
    
    Returns:
        Tuple of (prepared_features, pipeline)
        
    Raises:
        ValueError: If fit_pipeline=False and preprocessing_pipeline is None
    """
    # Create pipeline if fitting from scratch
    if preprocessing_pipeline is None:
        if not fit_pipeline:
            raise ValueError(
                "preprocessing_pipeline must be provided when fit_pipeline=False"
            )
        preprocessing_pipeline = build_preprocessing_pipeline()
    
    # Apply preprocessing
    X_prepared, pipeline = apply_preprocessing_pipeline(
        X,
        preprocessing_pipeline,
        fit=fit_pipeline
    )
    
    return X_prepared, pipeline
