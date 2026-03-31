"""
Persistence module.

This module handles saving and loading of trained models and preprocessing
pipelines. Separating persistence from training/prediction logic keeps those
functions focused and testable.
"""

import joblib
import json
from pathlib import Path
from typing import Union, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import Config


def save_model(
    model: Union[RandomForestClassifier, LogisticRegression],
    model_path: str | Path = None
) -> Path:
    """
    Serialize and save trained model to disk.
    
    Parameters:
        model: Fitted model object to save
        model_path: Path where model should be saved. If None, uses Config default.
    
    Returns:
        Path object pointing to the saved model
    """
    if model_path is None:
        model_path = Config.MODEL_PATH
    
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path


def save_pipeline(
    pipeline: object,
    pipeline_path: str | Path = None
) -> Path:
    """
    Serialize and save preprocessing pipeline to disk.
    
    Parameters:
        pipeline: Fitted preprocessing pipeline to save
        pipeline_path: Path where pipeline should be saved. If None, uses Config default.
    
    Returns:
        Path object pointing to the saved pipeline
    """
    if pipeline_path is None:
        pipeline_path = Config.PIPELINE_PATH
    
    pipeline_path = Path(pipeline_path)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to {pipeline_path}")
    
    return pipeline_path


def save_artifacts(
    model: Union[RandomForestClassifier, LogisticRegression],
    pipeline: object,
    model_path: str | Path = None,
    pipeline_path: str | Path = None
) -> Dict[str, Path]:
    """
    Save both model and pipeline artifacts.
    
    Parameters:
        model: Fitted model object
        pipeline: Fitted preprocessing pipeline
        model_path: Path for model file
        pipeline_path: Path for pipeline file
    
    Returns:
        Dictionary with keys 'model' and 'pipeline' pointing to saved files
    """
    model_saved = save_model(model, model_path)
    pipeline_saved = save_pipeline(pipeline, pipeline_path)
    
    return {
        "model": model_saved,
        "pipeline": pipeline_saved
    }


def save_metrics(
    metrics: Dict[str, float],
    metrics_path: str | Path = None
) -> Path:
    """
    Save evaluation metrics to JSON file.
    
    Parameters:
        metrics: Dictionary of metric names to values
        metrics_path: Path where metrics should be saved
    
    Returns:
        Path object pointing to the saved metrics
    """
    if metrics_path is None:
        metrics_path = Config.REPORT_PATH.parent / "metrics.json"
    
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable values to strings
    metrics_serializable = {k: float(v) if v is not None else None for k, v in metrics.items()}
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    return metrics_path


def load_metrics(
    metrics_path: str | Path = None
) -> Dict[str, float]:
    """
    Load evaluation metrics from JSON file.
    
    Parameters:
        metrics_path: Path to metrics file
    
    Returns:
        Dictionary of metric names to values
    """
    if metrics_path is None:
        metrics_path = Config.REPORT_PATH.parent / "metrics.json"
    
    metrics_path = Path(metrics_path)
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    print(f"Metrics loaded from {metrics_path}")
    
    return metrics
