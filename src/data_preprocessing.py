"""
Data preprocessing module.

This module handles all data loading, cleaning, and splitting operations.
Following the single responsibility principle, each function performs
one conceptual task: loading, cleaning, or splitting data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import Config


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    This function reads a CSV file and returns a DataFrame with minimal
    transformation. It is responsible only for data ingestion, not validation
    or cleaning.
    
    Parameters:
        filepath: Path to the CSV file containing raw data
        
    Returns:
        DataFrame containing the raw data as read from file
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.ParserError: If the file is not a valid CSV
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    required_columns: list[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    This function addresses missing data according to a specified strategy.
    For numerical columns, it fills with median or mean. For categorical
    columns, it fills with the mode.
    
    Parameters:
        df: Input DataFrame potentially containing NaN values
        strategy: Strategy for handling missing values ("median", "mean", "drop")
        required_columns: List of column names that must not have missing values.
                         Raises ValueError if any are missing after handling.
    
    Returns:
        DataFrame with missing values handled according to strategy
        
    Raises:
        ValueError: If required_columns contain missing values after handling
    """
    df = df.copy()
    
    # Fill numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "median":
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    elif strategy == "mean":
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    elif strategy == "drop":
        df = df.dropna(subset=numerical_cols)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Validate required columns
    if required_columns:
        missing_in_required = df[required_columns].isnull().any()
        if missing_in_required.any():
            raise ValueError(
                f"Required columns still have missing values: {missing_in_required[missing_in_required].index.tolist()}"
            )
    
    return df


def remove_duplicates(df: pd.DataFrame, subset: list[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Parameters:
        df: Input DataFrame
        subset: Column names to consider when identifying duplicates.
               If None, all columns are used.
    
    Returns:
        DataFrame with duplicates removed, keeping the first occurrence
    """
    df = df.copy()
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    rows_removed = initial_rows - len(df)
    
    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate rows")
    
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split DataFrame into training and test sets.
    
    This function separates features from target and splits into train/test
    sets with stratification to maintain class distribution.
    
    Parameters:
        df: Complete DataFrame with features and target
        target_column: Name of the target column
        test_size: Proportion of data reserved for testing (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
        - X_train, X_test: Feature DataFrames
        - y_train, y_test: Target Series
        
    Raises:
        ValueError: If target_column not in DataFrame
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Use stratification if target is binary/categorical
    if y.nunique() <= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def clean_data(
    df: pd.DataFrame,
    target_column: str = None,
    missing_strategy: str = "median"
) -> pd.DataFrame:
    """
    Orchestrate full data cleaning pipeline.
    
    This function combines multiple cleaning operations in sequence:
    removes duplicates and handles missing values.
    
    Parameters:
        df: Raw input DataFrame
        target_column: Optional target column name for validation
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame ready for feature engineering and modeling
    """
    df = df.copy()
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle missing values
    required_cols = [target_column] if target_column else None
    df = handle_missing_values(df, strategy=missing_strategy, required_columns=required_cols)
    
    return df
