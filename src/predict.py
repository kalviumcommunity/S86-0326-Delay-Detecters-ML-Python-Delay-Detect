"""
Prediction module.

This module handles inference on new data. It loads fitted models and
preprocessing pipelines and generates predictions. Critically, it only
calls transform() on pipelines, never fit_transform(), to prevent data leakage.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import Config


def load_artifacts(
    model_path: str | Path = None,
    pipeline_path: str | Path = None
) -> Tuple[Union[RandomForestClassifier, LogisticRegression], object]:
    """
    Load saved model and preprocessing pipeline from disk.
    
    This function loads fitted artifacts that were saved during training.
    These artifacts enable reproducible inference without retraining.
    
    Parameters:
        model_path: Path to saved model pickle file. If None, uses Config default.
        pipeline_path: Path to saved preprocessing pipeline pickle file.
                      If None, uses Config default.
    
    Returns:
        Tuple of (loaded_model, loaded_pipeline)
        
    Raises:
        FileNotFoundError: If either file does not exist
    """
    if model_path is None:
        model_path = Config.MODEL_PATH
    if pipeline_path is None:
        pipeline_path = Config.PIPELINE_PATH
    
    model_path = Path(model_path)
    pipeline_path = Path(pipeline_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Loaded pipeline from {pipeline_path}")
    
    return model, pipeline


def preprocess_new_data(
    X_new: pd.DataFrame,
    pipeline: object
) -> np.ndarray:
    """
    Apply preprocessing pipeline to new data.
    
    CRITICAL: This uses transform(), not fit_transform(). The pipeline was
    fitted on training data during model training. Applying it to new data
    uses those learned parameters (e.g., scaler mean/std, encoder mappings).
    
    If this called fit_transform(), new data would be used to learn
    transformation parameters, causing data leakage and invalid predictions.
    
    Parameters:
        X_new: New raw features as DataFrame
        pipeline: Fitted preprocessing pipeline
    
    Returns:
        Transformed features as numpy array
        
    Raises:
        ValueError: If X_new has different columns than pipeline expects
    """
    # Apply trained transformations (only transform, never fit_transform)
    X_prepared = pipeline.transform(X_new)
    
    return X_prepared


def predict(
    X_new: pd.DataFrame,
    model: Union[RandomForestClassifier, LogisticRegression],
    pipeline: object,
    return_probabilities: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions on new data using loaded artifacts.
    
    This function orchestrates the inference pipeline: preprocess new data
    using the fitted pipeline, then generate predictions using the fitted model.
    
    Parameters:
        X_new: New raw feature data as DataFrame
        model: Fitted model object
        pipeline: Fitted preprocessing pipeline
        return_probabilities: If True, also return prediction probabilities
    
    Returns:
        If return_probabilities=False:
            Array of class predictions
        If return_probabilities=True:
            Tuple of (predictions, probabilities)
    """
    # Preprocess new data using fitted pipeline (transform only, not fit)
    X_prepared = preprocess_new_data(X_new, pipeline)
    
    # Generate predictions
    predictions = model.predict(X_prepared)
    
    if return_probabilities:
        try:
            probabilities = model.predict_proba(X_prepared)
            return predictions, probabilities
        except Exception:
            print("Warning: Model does not support predict_proba()")
            return predictions, None
    
    return predictions


def predict_with_confidence(
    X_new: pd.DataFrame,
    model: Union[RandomForestClassifier, LogisticRegression],
    pipeline: object
) -> pd.DataFrame:
    """
    Generate predictions with confidence scores.
    
    This function returns predictions along with confidence information,
    useful for understanding model certainty.
    
    Parameters:
        X_new: New raw feature data as DataFrame
        model: Fitted model object
        pipeline: Fitted preprocessing pipeline
    
    Returns:
        DataFrame with columns: prediction, confidence
    """
    # Preprocess
    X_prepared = preprocess_new_data(X_new, pipeline)
    
    # Get predictions and probabilities
    predictions = model.predict(X_prepared)
    
    try:
        probabilities = model.predict_proba(X_prepared)
        confidence = probabilities.max(axis=1)
    except Exception:
        confidence = np.ones(len(predictions))
    
    # Return as DataFrame
    result_df = pd.DataFrame({
        "prediction": predictions,
        "confidence": confidence
    })
    
    return result_df


def batch_predict(
    X_new_list: list[pd.DataFrame],
    model: Union[RandomForestClassifier, LogisticRegression],
    pipeline: object
) -> list[np.ndarray]:
    """
    Generate predictions on multiple batches of new data.
    
    Useful for large datasets that should be processed in chunks
    to manage memory.
    
    Parameters:
        X_new_list: List of DataFrames to generate predictions for
        model: Fitted model object
        pipeline: Fitted preprocessing pipeline
    
    Returns:
        List of prediction arrays corresponding to input batches
    """
    predictions_list = []
    
    for i, X_batch in enumerate(X_new_list):
        batch_predictions = predict(X_batch, model, pipeline)
        predictions_list.append(batch_predictions)
        print(f"Processed batch {i+1}/{len(X_new_list)}")
    
    return predictions_list
