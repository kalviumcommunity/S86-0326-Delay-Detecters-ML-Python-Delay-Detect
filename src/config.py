"""
Centralized configuration for the ML pipeline.

This module contains all configuration parameters including file paths,
random state, hyperparameters, and column specifications. By centralizing
configuration, we ensure reproducibility and make it easy to modify
parameters without touching function implementations.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "delivery_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_features.parquet"

# Model and artifact paths
MODEL_PATH = PROJECT_ROOT / "models" / "delay_risk_model.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"

# Report and log paths
REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation_summary.md"
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"

# Reproducibility
RANDOM_STATE = 42

# Data configuration
TARGET_COLUMN = "is_delayed"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature engineering
CATEGORICAL_COLS = ["zone", "day_of_week", "peak_hour"]
NUMERICAL_COLS = ["distance_km", "items_count", "order_value", "day_of_month"]

# Explicit exclusions (identifiers, post-outcome columns, raw timestamps, etc.)
EXCLUDED_COLUMNS = ["delivery_id", "created_at", "updated_at"]

# Combined feature list used throughout the pipeline
ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

# Model hyperparameters
MODEL_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

# Evaluation metrics to compute
EVAL_METRICS = ["precision", "recall", "f1", "roc_auc", "accuracy"]


class Config:
    """Configuration class for accessing and validating all settings."""

    # Paths
    RAW_DATA_PATH = RAW_DATA_PATH
    PROCESSED_DATA_PATH = PROCESSED_DATA_PATH
    MODEL_PATH = MODEL_PATH
    PIPELINE_PATH = PIPELINE_PATH
    REPORT_PATH = REPORT_PATH
    LOG_PATH = LOG_PATH

    # Data settings
    TARGET_COLUMN = TARGET_COLUMN
    TEST_SIZE = TEST_SIZE
    VALIDATION_SIZE = VALIDATION_SIZE
    RANDOM_STATE = RANDOM_STATE

    # Feature columns
    CATEGORICAL_COLS = CATEGORICAL_COLS
    NUMERICAL_COLS = NUMERICAL_COLS
    EXCLUDED_COLUMNS = EXCLUDED_COLUMNS
    ALL_FEATURES = ALL_FEATURES

    # Model settings
    MODEL_HYPERPARAMS = MODEL_HYPERPARAMS
    EVAL_METRICS = EVAL_METRICS

    @staticmethod
    def ensure_directories():
        """Create all required directories if they don't exist."""
        directories = [
            Config.RAW_DATA_PATH.parent,
            Config.PROCESSED_DATA_PATH.parent,
            Config.MODEL_PATH.parent,
            Config.REPORT_PATH.parent,
            Config.LOG_PATH.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
