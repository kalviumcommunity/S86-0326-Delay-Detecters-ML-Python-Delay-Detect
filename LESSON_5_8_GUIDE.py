"""
Lesson 5.8: Structuring Python Files and Modules for Model Code
Complete Guide and Reference

This document demonstrates professional module structuring for ML projects.
"""

# =============================================================================
# SECTION 1: FROM SCRIPTS TO MODULES - THE CONCEPTUAL SHIFT
# =============================================================================

"""
KEY PRINCIPLE: A Python file becomes a module when it's imported into another file.
Think of files as building blocks that expose clearly defined functionality.
"""

# ❌ ANTI-PATTERN: All logic at top level (executes on import)
# File: bad_train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')           # RUNS ON IMPORT
df = df.dropna()                        # RUNS ON IMPORT  
X = df.drop('target', axis=1)          # RUNS ON IMPORT
y = df['target']                        # RUNS ON IMPORT
model = RandomForestClassifier()        # RUNS ON IMPORT
model.fit(X, y)                         # RUNS ON IMPORT
print(model.score(X, y))                # RUNS ON IMPORT

# Problem: If another module tries to import from this file, all training
# logic runs immediately as an unintended side effect!


# ✅ GOOD PATTERN: Functions with controlled entry point
# File: good_train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data(filepath):
    """Load and return raw data from CSV."""
    return pd.read_csv(filepath)


def clean_data(df):
    """Remove missing values from DataFrame."""
    return df.dropna()


def split_data(df, target_column):
    """Split data into features and target."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def train_model(X, y):
    """Train and return a Random Forest model."""
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


def main():
    """Main execution logic."""
    df = load_data('data.csv')
    df = clean_data(df)
    X, y = split_data(df, 'target')
    model = train_model(X, y)
    print(f"Model score: {model.score(X, y):.3f}")


if __name__ == "__main__":
    # This only runs when the file is executed directly as a script,
    # NOT when it's imported as a module
    main()


# Benefits:
# ✅ Can import functions without triggering execution
# ✅ Functions are reusable
# ✅ Execution is explicit and controlled
# ✅ Code is testable in isolation


# =============================================================================
# SECTION 2: DIRECTORY STRUCTURE FOR ML PROJECTS
# =============================================================================

"""
PROFESSIONAL ML PROJECT STRUCTURE

project_root/
├── data/
│   ├── raw/                          # Original, immutable data
│   │   └── delivery_data.csv
│   └── processed/                    # Cleaned, transformed data
│
├── models/                           # Saved model artifacts
│   ├── delay_risk_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── reports/                          # Evaluation reports, plots
│   └── metrics.json
│
├── logs/                             # Experiment logs
│   └── pipeline.log
│
├── src/                              # Source code package
│   ├── __init__.py                   # Makes src a Python package
│   ├── config.py                     # Centralized configuration
│   ├── data_preprocessing.py         # Load, clean, split
│   ├── feature_engineering.py        # Encode, scale, transform
│   ├── train.py                      # Model training
│   ├── evaluate.py                   # Evaluation metrics
│   ├── predict.py                    # Inference
│   └── persistence.py                # Save/load artifacts
│
├── tests/                            # Unit tests
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_train.py
│
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Project documentation
└── main.py                           # Entry point orchestration
"""

# =============================================================================
# SECTION 3: DESIGNED MODULES WITH CLEAR RESPONSIBILITIES
# =============================================================================

"""
EACH MODULE ANSWERS ONE QUESTION CLEARLY

Module Responsibility:
├── data_preprocessing.py    → How do we clean and prepare raw data?
├── feature_engineering.py   → How do we transform cleaned data into features?
├── train.py                 → How do we fit a model on prepared features?
├── evaluate.py              → How do we measure model performance?
├── predict.py               → How do we generate predictions on new data?
├── persistence.py           → How do we save and load artifacts?
└── config.py                → What are all the shared configuration values?

PRINCIPLE: No overlapping responsibilities, no code duplication.
"""

# ✅ Example: Each module does ONE thing

# data_preprocessing.py
def load_data(filepath):
    """Load data - ONLY loads, nothing else"""
    import pandas as pd
    return pd.read_csv(filepath)


# feature_engineering.py
def encode_features(df, categorical_cols):
    """Encode features - ONLY encodes, nothing else"""
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# train.py
def train_model(X_train, y_train):
    """Train model - ONLY trains, nothing else"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


# evaluate.py
def evaluate_model(model, X_test, y_test):
    """Evaluate - ONLY computes metrics, never prints or saves"""
    return {'accuracy': model.score(X_test, y_test)}


# =============================================================================
# SECTION 4: MANAGING IMPORTS CORRECTLY
# =============================================================================

"""
IMPORT BEST PRACTICES

1. USE ABSOLUTE IMPORTS from project root
2. AVOID CIRCULAR IMPORTS through careful module hierarchy
3. AVOID WILDCARD IMPORTS that obscure source
4. KEEP DEPENDENCIES EXPLICIT and clear
"""

# ✅ GOOD: Absolute imports from project root
from src.config import RANDOM_STATE, DATA_PATH, MODEL_PATH
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_features

# ❌ BAD: Relative imports (fragile, unclear)
# from ..data_preprocessing import load_data

# ❌ BAD: Wildcard imports
# from src.data_preprocessing import *

# ✅ DEPENDENCY HIERARCHY (prevents circular imports)
"""
                    ↑ depends on ↓
             config.py (shared constants)
                    ↑
            utils.py (shared utilities)
                    ↑
        data_preprocessing.py (loads, cleans)
                    ↑
        feature_engineering.py (transforms)
                    ↑
            train.py (fits model)
                    ↑
         evaluate.py (computes metrics)
                    ↑
           predict.py (generates predictions)

Higher-level modules depend on lower-level ones, NEVER vice versa.
This creates a directed acyclic graph (DAG) - no cycles possible!
"""


# =============================================================================
# SECTION 5: ENCAPSULATING MODEL CODE PROPERLY
# =============================================================================

"""
CLEAN ENCAPSULATION PRINCIPLES

✅ Functions accept ALL inputs as parameters
✅ Functions return outputs explicitly  
✅ No global variables or hidden state
✅ Clear documentation of behavior
✅ Functions are testable in isolation
"""

# ✅ CLEAN ENCAPSULATION: Explicit inputs and outputs
def train_model_clean(X_train, y_train, learning_rate=0.1, random_state=42):
    """
    Train a model on provided data.
    
    Parameters:
        X_train: Training features (required)
        y_train: Training labels (required)
        learning_rate: Learning rate (configurable)
        random_state: Random seed (explicit)
    
    Returns:
        Fitted model object
    
    Benefits:
    - All inputs visible in signature
    - No hidden dependencies
    - Easily testable
    - Easy to change behavior
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ❌ POOR ENCAPSULATION: Depends on global variables
RANDOM_STATE = 42  # Global variable
X_TRAIN_GLOBAL = None  # Global state


def train_model_bad():
    """
    Train a model - BAD way.
    
    Problems:
    - Where does X_TRAIN come from? Not in signature!
    - Where does y_train come from? Not in signature!
    - Can't test without setting globals first
    - Hard to debug when globals change
    - Tightly coupled to environment
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_TRAIN_GLOBAL, y_train)  # y_train undefined!
    return model


# ✅ PIPELINE ENCAPSULATION: Explicit pipeline building
def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Build a preprocessing pipeline.
    
    Returns the pipeline object explicitly - not created in global scope.
    This makes it reusable, testable, and portable.
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])

    return preprocessor


# =============================================================================
# SECTION 6: SEPARATING TRAINING AND PREDICTION CODE
# =============================================================================

"""
CRITICAL SEPARATION: Training vs Prediction

Training Pipeline:
- Load RAW data
- Split into train/test  
- Fit preprocessing ON TRAINING DATA ONLY
- Fit model ON TRAINING DATA ONLY
- Evaluate on test data
- Save fitted artifacts

Prediction Pipeline:
- Load ALREADY-FITTED artifacts
- Validate input schema
- Apply preprocessing using .transform() (NOT .fit_transform()!)
- Generate predictions
- Return predictions

✅ PRINCIPLE: Prediction NEVER fits, NEVER learns from new data
❌ If prediction.py calls fit_transform(), it's data leakage!
"""

# ✅ CORRECT: train.py calls fit_transform() on training data
def train_correct(X_train, y_train, X_test, y_test):
    """
    Training: FIT preprocessing on training data.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Fit preprocessing ON TRAINING DATA
    pipeline = build_preprocessing_pipeline(['cat_col'], ['num_col'])
    X_train_processed = pipeline.fit_transform(X_train)  # ✅ FIT here
    
    # Fit model ON TRAINING DATA  
    model = RandomForestClassifier()
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    X_test_processed = pipeline.transform(X_test)  # ✅ TRANSFORM only
    score = model.score(X_test_processed, y_test)
    
    # Save artifacts for later reuse
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(pipeline, 'models/pipeline.pkl')
    
    return model, pipeline


# ✅ CORRECT: predict.py only calls transform()
def predict_correct(new_data):
    """
    Prediction: LOAD fitted artifacts, apply with transform() only.
    """
    import joblib
    
    # Load already-fitted artifacts
    model = joblib.load('models/model.pkl')
    pipeline = joblib.load('models/pipeline.pkl')
    
    # TRANSFORM using FITTED pipeline (NOT fit_transform!)
    new_data_processed = pipeline.transform(new_data)  # ✅ TRANSFORM only
    
    # Generate predictions
    predictions = model.predict(new_data_processed)
    
    return predictions


# ❌ WRONG: predict.py refits preprocessing
def predict_wrong(new_data):
    """
    Prediction: WRONG - Refits preprocessing!
    This is data leakage!
    """
    # ❌ WRONG: This learns from new_data to transform new_data
    new_data_processed = pipeline.fit_transform(new_data)
    
    # If we learn scaling from new_data then use it to transform new_data,
    # we're leaking information about new_data into the transformation!
    predictions = model.predict(new_data_processed)
    return predictions


# =============================================================================
# SECTION 7: CONFIGURATION MANAGEMENT
# =============================================================================

"""
CENTRALIZE ALL CONFIGURATION IN ONE FILE

Benefits:
✅ Single source of truth
✅ Easy to change parameters
✅ Changes propagate automatically
✅ Explicit and reviewable
✅ No magic numbers scattered in code
"""

# config.py
"""
Centralized configuration for the ML project.
ALL constants, file paths, and hyperparameters defined here.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Data paths (relative to project root)
RAW_DATA_PATH = "data/raw/delivery_data.csv"
PROCESSED_DATA_PATH = "data/processed/delivery_features.parquet"

# Model artifact paths
MODEL_PATH = "models/delay_risk_model.pkl"
PIPELINE_PATH = "models/preprocessing_pipeline.pkl"

# Report paths
REPORT_PATH = "reports/evaluation_summary.md"
LOG_PATH = "logs/pipeline.log"

# Train/test split
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Model hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# Feature columns
TARGET_COLUMN = "is_delayed"
CATEGORICAL_COLS = ["zone", "day_of_week", "peak_hour"]
NUMERICAL_COLS = ["distance_km", "items_count", "order_value"]

# Example: Now every module imports from config
# Inside train.py:
# from src.config import RANDOM_STATE, PIPELINE_PATH, MODEL_PATH
# 
# Inside predict.py:
# from src.config import MODEL_PATH, PIPELINE_PATH
#
# Inside data_preprocessing.py:
# from src.config import RAW_DATA_PATH, TEST_SIZE


# =============================================================================
# SECTION 8: TESTING AND MAINTAINABILITY
# =============================================================================

"""
MODULAR STRUCTURE ENABLES UNIT TESTING

When functions are small, focused, and free of side effects,
they can be tested in isolation.

Without modular structure:
- Can't test individual components
- Must test entire system end-to-end
- Failures are hard to debug
- Refactoring is risky

With modular structure:
- Test each function independently
- Failures are localized
- Debugging is fast
- Refactoring is safe
"""

# ============================================================================
# SECTION 9: COMMON STRUCTURAL MISTAKES AND SOLUTIONS  
# =============================================================================

"""
MISTAKE 1: ALL LOGIC IN A SINGLE FILE

❌ Problem:
- 2000+ line file with everything mixed together
- Can't isolate a single component to test
- Hard to find relevant code when debugging
- Impossible to reuse preprocessing

✅ Solution:
- Separate into focused modules
- One responsibility per file
"""

# ❌ BAD: all_in_one.py (2000 lines)
# def main():
#     # Load data
#     df = pd.read_csv('data.csv')
#     
#     # Clean data 
#     df = df.dropna()
#     
#     # Preprocess
#     df = pd.get_dummies(df)
#     X = df.drop('target', axis=1)
#     y = df['target']
#     
#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#     
#     # Train
#     model = RandomForest()
#     model.fit(X_train, y_train)
#     
#     # Evaluate
#     print(f"Accuracy: {model.score(X_test, y_test)}")
#     
#     # Plot results
#     plt.figure()
#     # ... plotting code ...
#     
#     # Save model
#     joblib.dump(model, 'model.pkl')

# ✅ GOOD: Separate modules
# data_preprocessing.py: load_data(), clean_data(), split_data()
# feature_engineering.py: encode_features(), scale_features()
# train.py: train_model()
# evaluate.py: evaluate_model()
# persistence.py: save_model()


"""
MISTAKE 2: DUPLICATE PREPROCESSING CODE IN TRAINING AND PREDICTION

❌ Problem:
- Copy-paste preprocessing code into both train.py and predict.py
- Two versions of the same logic
- When you fix a bug in one, you forget to fix the other
- Eventually they diverge
- Predictions don't match training behavior

✅ Solution:
- Write preprocessing ONCE in feature_engineering.py
- Save fitted pipeline as artifact
- Load and reuse in BOTH training and prediction
"""

# ❌ BAD: Duplicate code
# train.py
# df['category'] = df['category'].map({'A': 1, 'B': 2, 'C': 3})
# df['numeric'] = (df['numeric'] - df['numeric'].mean()) / df['numeric'].std()

# predict.py  
# df['category'] = df['category'].map({'A': 1, 'B': 2, 'C': 3})
# df['numeric'] = (df['numeric'] - df['numeric'].mean()) / df['numeric'].std()

# ✅ GOOD: Single source of truth
# feature_engineering.py
# def build_pipeline():
#     return ColumnTransformer([...])

# train.py
# pipeline = build_pipeline()
# X_train_processed = pipeline.fit_transform(X_train)

# predict.py
# pipeline = joblib.load('models/pipeline.pkl')
# X_new_processed = pipeline.transform(X_new)


"""
MISTAKE 3: HARDCODED ABSOLUTE FILE PATHS

❌ Problem:
df = pd.read_csv("/Users/yourname/Desktop/ML_project/data/telco_churn.csv")

- Works only on YOUR machine
- Doesn't work for teammates
- Doesn't work in production
- Path changes break everything

✅ Solution:
- Use relative paths from project root
- Centralize in config.py
"""

# ❌ BAD
# df = pd.read_csv("/c/Users/Admin/Desktop/ML-python/data/raw/delivery_data.csv")

# ✅ GOOD
# config.py
# DATA_PATH = "data/raw/delivery_data.csv"

# Any module can now use:
# from src.config import DATA_PATH
# df = pd.read_csv(DATA_PATH)


"""
MISTAKE 4: CIRCULAR IMPORTS BETWEEN MODULES

❌ Problem:
- train.py imports from evaluate.py
- evaluate.py imports from train.py
- Python can't resolve this
- ImportError at runtime

✅ Solution:
- Move shared constants to config.py
- Move shared utilities to utils.py
- Enforce dependency hierarchy
"""

# ❌ BAD: Circular import
# train.py
# from src.evaluate import compute_metrics
# from src import evaluate

# evaluate.py
# from src.train import train_model  # Circular!

# ✅ GOOD: Use shared config/utils
# config.py: RANDOM_STATE = 42

# train.py
# from src.config import RANDOM_STATE

# evaluate.py
# from src.config import RANDOM_STATE


"""
MISTAKE 5: NO CLEAR ENTRY POINT FOR EXECUTION

❌ Problem:
- Code scattered at top level executes in unpredictable order
- Different behavior depending on import order
- No clear "main" orchestration point

✅ Solution:
- Use if __name__ == "__main__": pattern
- Define clear main() function
- Make execution explicit
"""

# ❌ BAD: No clear entry point
# train.py
# df = load_data('data.csv')
# model = train_model(df)
# print(model.score(...))  # Runs on import!

# ✅ GOOD: Controlled entry point
# train.py
# def main():
#     df = load_data('data.csv')
#     model = train_model(df)
#     print(model.score(...))

# if __name__ == "__main__":
#     main()  # Only runs when file is executed directly


"""
MISTAKE 6: HIDDEN GLOBAL VARIABLES CONTROLLING BEHAVIOR

❌ Problem:
- Functions depend on values set elsewhere
- Not visible in function signature
-Hard to test
- Fragile

✅ Solution:
- Pass configuration as parameters
- Import from config.py explicitly
- Make dependencies visible
"""

# ❌ BAD: Hidden global dependency
# RANDOM_STATE = 42  # Somewhere at top of file
#
# def train_model(X, y):
#     model = RandomForest(random_state=RANDOM_STATE)  # Where does this come from?
#     model.fit(X, y)
#     return model

# ✅ GOOD: Explicit parameter
# from src.config import RANDOM_STATE
#
# def train_model(X, y, random_state=None):
#     if random_state is None:
#         random_state = RANDOM_STATE
#     model = RandomForest(random_state=random_state)
#     model.fit(X, y)
#     return model


print("=" * 80)
print("LESSON 5.8: Structuring Python Files and Modules for Model Code")
print("=" * 80)
print("\nKey Takeaways:")
print("✅ Structure is more important than algorithms")
print("✅ Use functions with controlled entry points")
print("✅ Separate concerns into focused modules")
print("✅ Centralize configuration")
print("✅ Prevent data leakage by separating training and prediction")
print("✅ Avoid circular imports and hidden global state")
print("✅ Write testable, reusable code")
print("\nYour project in d:/ML-python demonstrates ALL these principles!")
print("=" * 80)
