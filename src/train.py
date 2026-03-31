"""
Model training module.

This module handles model instantiation and training. It receives prepared
feature data and a target, and returns a trained model artifact.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union

from src.config import Config


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    random_state: int = None,
    **hyperparams
) -> Union[RandomForestClassifier, LogisticRegression]:
    """
    Train a machine learning model on training data.
    
    This function receives prepared training data and trains a model according
    to specified type and hyperparameters. It is responsible only for training,
    not for data preparation or evaluation.
    
    The function returns the fitted model artifact, which is then saved by the
    orchestration layer. This separation ensures that training logic is distinct
    from persistence logic.
    
    Parameters:
        X_train: Training feature array (prepared by feature engineering)
        y_train: Training target array
        model_type: Type of model ("random_forest" or "logistic_regression")
        random_state: Random seed for reproducibility. If None, uses Config default.
        **hyperparams: Additional model-specific hyperparameters
    
    Returns:
        Fitted model object
        
    Raises:
        ValueError: If model_type is not supported
        ValueError: If X_train and y_train have incompatible lengths
    """
    if random_state is None:
        random_state = Config.RANDOM_STATE
    
    # Validate input
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have same length. Got {len(X_train)} and {len(y_train)}"
        )
    
    # Instantiate model
    if model_type == "random_forest":
        # Use config defaults, allow overrides via hyperparams
        params = Config.MODEL_HYPERPARAMS.copy()
        params.update(hyperparams)
        model = RandomForestClassifier(**params)
    
    elif model_type == "logistic_regression":
        params = {"random_state": random_state, "max_iter": 1000}
        params.update(hyperparams)
        model = LogisticRegression(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    print(f"Training {model_type} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    print("Training complete.")
    
    return model


def train_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "random_forest",
    random_state: int = None,
    **hyperparams
) -> dict:
    """
    Train a model and evaluate on validation set during training.
    
    This function trains a model and tracks validation performance, useful
    for early stopping and hyperparameter tuning decisions.
    
    Parameters:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_type: Type of model to train
        random_state: Random seed
        **hyperparams: Model hyperparameters
    
    Returns:
        Dictionary containing:
        - 'model': Fitted model object
        - 'train_score': Training set score
        - 'val_score': Validation set score
    """
    model = train_model(
        X_train, y_train,
        model_type=model_type,
        random_state=random_state,
        **hyperparams
    )
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Training score: {train_score:.4f}")
    print(f"Validation score: {val_score:.4f}")
    
    return {
        "model": model,
        "train_score": train_score,
        "val_score": val_score
    }
