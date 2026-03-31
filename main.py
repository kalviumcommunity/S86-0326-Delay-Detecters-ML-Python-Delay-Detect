"""
Main orchestration script for the ML pipeline.

This script orchestrates the complete machine learning workflow:
1. Load raw data
2. Clean and preprocess
3. Split into train/test sets
4. Prepare features (feature engineering)
5. Train model
6. Evaluate on test set
7. Save artifacts for later inference
8. Generate predictions on new data

This separation makes the workflow readable, testable, and reproducible.
Individual functions can be tested in isolation, and the entire pipeline
can be recreated deterministically.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging

from src.config import Config
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import prepare_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts, save_metrics
from src.predict import predict


def setup_logging():
    """Configure logging for the pipeline."""
    Config.ensure_directories()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_PATH),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Execute the complete ML pipeline."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("ML PIPELINE EXECUTION START")
    logger.info("=" * 80)
    
    try:
        # ========== DATA LOADING ==========
        logger.info("Step 1: Loading raw data...")
        df = load_data(Config.RAW_DATA_PATH)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        logger.info(f"Columns: {list(df.columns)}")
        
        # ========== DATA CLEANING ==========
        logger.info("\nStep 2: Cleaning data...")
        df_clean = clean_data(df, target_column=Config.TARGET_COLUMN, missing_strategy="median")
        logger.info(f"Cleaned data: {len(df_clean)} samples")
        
        # ========== TRAIN/TEST SPLIT ==========
        logger.info("\nStep 3: Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(
            df_clean,
            target_column=Config.TARGET_COLUMN,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        logger.info(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        # ========== FEATURE ENGINEERING ==========
        logger.info("\nStep 4: Engineering features...")
        X_train_prepared, pipeline = prepare_features(
            X_train,
            fit_pipeline=True
        )
        X_test_prepared, _ = prepare_features(
            X_test,
            fit_pipeline=False,
            preprocessing_pipeline=pipeline
        )
        logger.info(f"Prepared features: {X_train_prepared.shape[1]} features")
        
        # ========== MODEL TRAINING ==========
        logger.info("\nStep 5: Training model...")
        model = train_model(
            X_train_prepared,
            y_train.values,
            model_type="random_forest",
            random_state=Config.RANDOM_STATE
        )
        logger.info("Model training complete")
        
        # ========== MODEL EVALUATION ==========
        logger.info("\nStep 6: Evaluating model on test set...")
        metrics = evaluate_model(
            model,
            X_test_prepared,
            y_test.values,
            metrics=Config.EVAL_METRICS
        )
        
        logger.info("Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name}: (not applicable)")
        
        # ========== SAVING ARTIFACTS ==========
        logger.info("\nStep 7: Saving model and preprocessing artifacts...")
        save_artifacts(model, pipeline)
        save_metrics(metrics)
        logger.info("Artifacts saved successfully")
        
        # ========== INFERENCE ON NEW DATA ==========
        logger.info("\nStep 8: Demonstrating inference on new data...")
        # Use a sample of test data as "new" data
        new_data_sample = X_test.head(5)
        predictions = predict(new_data_sample, model, pipeline, return_probabilities=True)
        
        if isinstance(predictions, tuple):
            preds, probs = predictions
            logger.info(f"Generated predictions for {len(preds)} new samples")
            logger.info(f"Sample predictions: {preds[:5]}")
            if probs is not None:
                logger.info(f"Sample probabilities: {probs[:5]}")
        else:
            logger.info(f"Generated predictions for {len(predictions)} new samples")
            logger.info(f"Sample predictions: {predictions[:5]}")
        
        logger.info("\n" + "=" * 80)
        logger.info("ML PIPELINE EXECUTION SUCCESS")
        logger.info("=" * 80)
        
        return {
            "model": model,
            "pipeline": pipeline,
            "metrics": metrics,
            "data": {
                "X_train": X_train_prepared,
                "X_test": X_test_prepared,
                "y_train": y_train,
                "y_test": y_test
            }
        }
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
