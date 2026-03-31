# ML Pipeline: Modularized, Production-Ready Architecture

## 5.7 Assignment: Refactoring Python Functions and Imports for ML Workflows

This project demonstrates **professional ML engineering principles** through a modular, reusable pipeline architecture. It transforms fragmented notebook code into a structured system that is reproducible, testable, and deployable.

---

## Project Structure

```
ML-python/
├── data/
│   ├── raw/                           # Immutable source data
│   │   └── delivery_data.csv
│   └── processed/                     # Cleaned, transformed data
├── models/                            # Saved model artifacts
│   ├── delay_risk_model.pkl           # Fitted model
│   └── preprocessing_pipeline.pkl     # Fitted transformations
├── reports/                           # Evaluation reports and metrics
│   └── evaluation_summary.md
├── logs/                              # Pipeline execution logs
├── src/                               # Core pipeline modules (Lesson 5.7)
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration
│   ├── data_preprocessing.py          # Load, clean, split data
│   ├── feature_engineering.py         # Encode, scale, transform features
│   ├── train.py                       # Model training
│   ├── evaluate.py                    # Model evaluation
│   ├── predict.py                     # Inference on new data
│   └── persistence.py                 # Save/load artifacts
├── tests/                             # Unit tests (Lesson 5.8)
│   ├── __init__.py
│   ├── test_preprocessing.py          # 7 tests for data module
│   ├── test_feature_engineering.py    # 8 tests for features module
│   └── test_train.py                  # 8 tests for training module
├── main.py                            # Orchestration script
├── create_sample_data.py              # Generate test data
├── LESSON_5_8_GUIDE.py                # Comprehensive lesson on module structuring (550+ lines)
├── LESSON_5_8_ASSIGNMENT.md           # Lesson 5.8 assignment guide (400+ lines)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Core Design Principles

### 1. Single Responsibility Principle
Each function performs **exactly one conceptual task**:
- **`data_preprocessing.py`**: Loading, cleaning, and splitting data
- **`feature_engineering.py`**: Encoding, scaling, and feature transformation
- **`train.py`**: Model instantiation and training
- **`evaluate.py`**: Evaluation metrics computation
- **`predict.py`**: Inference on new data
- **`persistence.py`**: Saving and loading artifacts

### 2. Clear Input/Output Contracts
All functions use:
- **Type hints** for parameters and return types
- **Comprehensive docstrings** explaining purpose, parameters, and return values
- **Explicit parameters** instead of relying on global state
- **Structured returns** (dictionaries, tuples) instead of printing

### 3. Configuration Centralization (`config.py`)
All configuration lives in one place:
- File paths (data, models, reports)
- Random states for reproducibility
- Hyperparameters
- Feature column names

This makes it trivial to modify parameters without touching function implementations.

### 4. Data Leakage Prevention
The pipeline enforces proper train/test isolation:
- **Training** uses `fit_transform()` to learn transformations from training data
- **Inference** uses `transform()` to apply fitted transformations without refitting
- This is enforced by function signatures and separation of concerns

---

## Module Descriptions

### `config.py`
Centralized configuration for the entire pipeline. Every path, parameter, and column specification is defined here.

**Key Features:**
- `Config` class with static methods for directory creation
- `ensure_directories()` creates output folders on demand
- All hyperparameters and evaluation metrics centralized

### `data_preprocessing.py`
Data loading, cleaning, and train/test splitting.

**Functions:**
- `load_data()` - Read CSV files
- `handle_missing_values()` - Fill NaNs with configurable strategy
- `remove_duplicates()` - Remove duplicate rows
- `split_data()` - Stratified train/test split with validation
- `clean_data()` - Orchestrator combining multiple cleaning steps

### `feature_engineering.py`
Feature encoding, scaling, and pipeline construction.

**Functions:**
- `encode_categorical_features()` - One-hot encode categorical variables
- `scale_numerical_features()` - Standardize numerical features
- `create_derived_features()` - Engineer new features from existing ones
- `build_preprocessing_pipeline()` - Construct sklearn ColumnTransformer
- `apply_preprocessing_pipeline()` - Apply fitted pipeline with explicit fit/transform control
- `prepare_features()` - Orchestrator combining all feature engineering steps

**CRITICAL:** Separates `fit=True` (training) from `fit=False` (inference) to prevent data leakage.

### `train.py`
Model instantiation and training.

**Functions:**
- `train_model()` - Fit model on training data, return artifact
- `train_with_validation()` - Train and track validation performance

**Key Features:**
- Receives prepared feature data (no preprocessing inside function)
- Returns fitted model object (not saved by training function)
- Supports multiple model types via configuration

### `evaluate.py`
Model evaluation and metrics computation.

**Functions:**
- `evaluate_model()` - Compute multiple metrics, return dictionary
- `compute_confusion_matrix()` - Generate confusion matrix
- `get_classification_report()` - Detailed per-class metrics
- `compare_metrics()` - Compare current vs baseline performance

**Key Features:**
- Returns metrics as structured data, doesn't print
- Supports multiple metric types through configuration
- Enables programmatic metric aggregation and logging

### `predict.py`
Inference on new data using saved artifacts.

**Functions:**
- `load_artifacts()` - Load saved model and pipeline
- `preprocess_new_data()` - Apply fitted transformations (transform only, never fit)
- `predict()` - Generate predictions on new data
- `predict_with_confidence()` - Return predictions with confidence scores
- `batch_predict()` - Process multiple batches of data

**CRITICAL:** Only calls `transform()` on preprocessing pipeline, never `fit_transform()`, preventing data leakage during inference.

### `persistence.py`
Saving and loading model artifacts.

**Functions:**
- `save_model()` - Serialize fitted model to pickle
- `save_pipeline()` - Serialize preprocessing pipeline
- `save_artifacts()` - Save both model and pipeline together
- `save_metrics()` - Save metrics to JSON
- `load_metrics()` - Load metrics from JSON

---

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Sample Data
```bash
python create_sample_data.py
```

This generates synthetic delivery data for testing. Replace with your own data in `data/raw/delivery_data.csv`.

### 3. Run Complete Pipeline
```bash
python main.py
```

This executes:
1. Load and clean data
2. Split into train/test sets
3. Engineer features
4. Train model
5. Evaluate on test set
6. Save artifacts
7. Demonstrate inference

Execution logs are saved to `logs/pipeline.log`.

### 4. Use Model for New Predictions
```python
from src.predict import load_artifacts, predict
import pandas as pd

# Load saved model and pipeline
model, pipeline = load_artifacts()

# Load new data
new_data = pd.read_csv("path/to/new_data.csv")

# Generate predictions
predictions = predict(new_data, model, pipeline)
```

---

## Testing and Quality Assurance (Lesson 5.8)

This project includes a comprehensive test suite demonstrating the **testability benefits of modular structure**.

### Running Unit Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests for specific module
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_train.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Suite Overview

- **23 total tests** - All passing ✅
- **test_preprocessing.py** (7 tests) - Data loading, cleaning, splitting
- **test_feature_engineering.py** (8 tests) - Encoding, scaling, pipeline construction
- **test_train.py** (8 tests) - Model training and validation

### Why Modular Structure Enables Testing

```python
# Can test preprocessing ALONE without running entire pipeline
from src.data_preprocessing import handle_missing_values
df_clean = handle_missing_values(df_with_nulls)
assert df_clean.isnull().sum().sum() == 0

# Can test feature engineering ALONE
from src.feature_engineering import encode_categorical_features
X_encoded = encode_categorical_features(X)
assert X_encoded.shape[0] == X.shape[0]

# Can test training ALONE with synthetic data
from src.train import train_model
model = train_model(X_train_synthetic, y_train_synthetic)
assert model is not None
```

### Key Testing Patterns

1. **Isolated Function Testing** - Each function tested independently
2. **Integration Testing** - Functions tested together in realistic workflows
3. **Data Validation** - Tests verify shapes, types, and value ranges
4. **Reproducibility** - Tests use fixed random states

---

## Lesson Materials

### Lesson 5.7: Reviewing Python Functions and Imports for ML Workflows
This project itself is the assignment—demonstrating:
- Clean function design with type hints and docstrings
- Proper import management avoiding circular dependencies
- Reusable functions at both training and inference time
- Production-ready ML pipeline architecture

### Lesson 5.8: Structuring Python Files and Modules for Model Code
Additional learning materials included:

- **`LESSON_5_8_GUIDE.py`** (550+ lines)
  - 9 comprehensive sections on module structuring
  - 40+ code examples showing best practices and anti-patterns
  - Covers: directory structure, module responsibilities, import management, data leakage prevention, testing patterns
  
- **`LESSON_5_8_ASSIGNMENT.md`** (400+ lines)
  - Maps all lesson concepts to this project
  - Shows how the delivery delay pipeline implements each principle
  - Includes implementation recommendations and checklists

---

## Key Engineering Principles Demonstrated

### ✅ Reproducibility
- **Random state** controlled in all operations
- **Configuration** centralized and version-controlled
- **Artifacts** saved for exact reproduction without retraining

### ✅ Modularity
- Each function has **single responsibility**
- Functions **independently testable**
- **No hidden dependencies** through global state
- **Clean imports** of only required functions

### ✅ Separation of Concerns
- **Data pipeline** (load → clean → split) separate from **ML pipeline** (train → evaluate)
- **Training** separate from **inference**
- **Preprocessing** strictly isolated from **model logic**

### ✅ Production Readiness
- **Artifact persistence** enables deployment without retraining
- **Proper error handling** and validation
- **Logging** tracks execution and issues
- **Configuration** externalizable for different environments

### ✅ Data Integrity
- **Train/test isolation** enforced by function signatures
- **Data leakage prevention** through fit/transform separation
- **Validation** of required columns and data shapes
- **Documented assumptions** in docstrings

---

## Common Patterns

### Pattern 1: Training with Config
```python
from src.config import Config
from src.train import train_model

model = train_model(
    X_train, y_train,
    random_state=Config.RANDOM_STATE
)
```

### Pattern 2: Feature Processing (Training)
```python
from src.feature_engineering import prepare_features

X_train_prepared, pipeline = prepare_features(
    X_train,
    fit_pipeline=True  # Fit transformations on training data
)
```

### Pattern 3: Feature Processing (Inference)
```python
X_test_prepared, _ = prepare_features(
    X_test,
    fit_pipeline=False,  # Do NOT fit, only apply
    preprocessing_pipeline=pipeline  # Use training pipeline
)
```

### Pattern 4: Evaluation and Metrics
```python
from src.evaluate import evaluate_model

metrics = evaluate_model(model, X_test, y_test)

# Metrics are a dictionary, not printed output
print(f"F1 Score: {metrics['f1']:.4f}")
```

### Pattern 5: Inference on New Data
```python
from src.predict import load_artifacts, predict

model, pipeline = load_artifacts()
predictions = predict(new_data, model, pipeline)
# Process predictions programmatically
results_df = new_data.copy()
results_df['prediction'] = predictions
```

---

## Why This Structure Matters

### Without Modularization
```
(Fragile notebook code)
↓
Can only run in sequence
Cannot debug individual components
Cannot reuse preprocessing at inference time
Cannot be version controlled effectively
Cannot collaborate
Cannot deploy
```

### With Modularization
```
(Clean, separate modules)
↓
Each component independently testable
Preprocessing reused identically at training and inference
Clear responsibility boundaries
Version controlled and collaborative
Production-ready and deployable
```

---

## Testing Individual Modules

Each module can be tested independently:

```python
# Test preprocessing
from src.data_preprocessing import handle_missing_values
import pandas as pd

df = pd.DataFrame({"a": [1, 2, None], "b": [None, "x", "y"]})
df_clean = handle_missing_values(df)
assert df_clean.isnull().sum().sum() == 0

# Test evaluation
from src.evaluate import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
assert metrics['f1'] > 0
assert 0 <= metrics['precision'] <= 1
```

---

## Extending the Pipeline

To add new components:

1. **New data source?** → Add function to `data_preprocessing.py`
2. **New feature?** → Add function to `feature_engineering.py`
3. **New model type?** → Add training logic to `train.py`
4. **New metric?** → Add to `evaluate.py` and update `config.py`
5. **Custom orchestration?** → Create new orchestration script, import from `src/`

All existing code remains unchanged and tested.

---

## References

- **Single Responsibility Principle**: Each function does one thing, does it well, changes for one reason
- **Separation of Concerns**: Training, evaluation, and inference logic cleanly separated
- **Configuration Management**: All parameters externalized for easy modification
- **Data Leakage Prevention**: Strict `fit()` vs `transform()` boundaries
- **Reproducibility**: Random states, centralized configuration, artifact persistence

This separation of concerns is good engineering practice because each stage can be tested and maintained independently.

---

## Workflow Mapping Through the Code

The ML workflow in this repository maps as:

Data -> Features -> Model -> Evaluation

### 1. Data Stage
- Source: raw order and delivery logs with timestamps and location metadata.
- Code location: preprocessing script.
- Typical operations:
	- Parse timestamp columns into datetime format.
	- Remove impossible records (delivery before order time).
	- Standardize zone names and city labels.
	- Handle null values in timestamps or location fields.

### 2. Feature Stage
- Code location: feature engineering script.
- Example features for this problem:
	- delivery_duration_minutes = delivered_time - order_time
	- is_peak_hour (binary)
	- day_of_week
	- zone_average_delay_last_7_days
	- rider_load_in_last_hour
- Why this matters: these features convert raw logs into numerical signals that capture operational behavior, not just raw timestamps.

### 3. Model Stage
- Code location: training script.
- Typical approach:
	- Choose a model suitable for tabular operational data (for example gradient boosting or random forest).
	- Split training and validation sets.
	- Fit model and store trained artifact in models.
- Output: reusable model file, not just notebook metrics.

### 4. Evaluation Stage
- Code location: evaluation script + report file.
- Typical metrics:
	- Regression target (delay minutes): MAE and RMSE.
	- Classification target (delayed or not): precision, recall, F1, ROC-AUC.
- Expected review check:
	- Metrics must be computed on unseen data.
	- Results should be compared to a baseline such as median-delay predictor.

This trace confirms where each pipeline stage lives and how data transitions from raw logs to measurable performance outcomes.

---

## One Specific Strength

### Strength: Clear pipeline modularity
The project separates preprocessing, feature engineering, training, and evaluation into dedicated scripts instead of mixing everything into one notebook.

Why this is strong:
- Prevents hidden coupling between steps.
- Makes debugging faster when delays spike in production.
- Enables consistent reuse of transformations at inference time.
- Supports team collaboration because each module has a clear responsibility.

This is a sign of production-minded project design.

---

## One Specific Weakness and Improvement Opportunity

### Weakness: Risk of temporal leakage
A common weakness in delivery-delay projects is performing random train-test splits on timestamped data, which can leak future behavior into training.

Why this is a problem:
- Delivery behavior changes over time (seasonality, weather, staffing changes).
- Random splitting can make evaluation look better than real-world performance.
- Deployment accuracy may drop because the model has effectively seen future patterns.

How to fix:
- Use time-based splits (train on earlier period, validate on later period).
- Fit all preprocessing statistics on training period only.
- Add backtesting by weekly or monthly windows.
- Report metrics by time segment and by zone, not only global averages.

This improvement directly increases trustworthiness of evaluation results.

---

## Red Flags I Would Check During Review

- No clear train-test separation for time-dependent records.
- Preprocessing fit on full dataset before split.
- Delay feature definitions missing or ambiguous.
- Hardcoded local file paths that break on other machines.
- No saved model artifact or no documented inference path.
- Metrics reported without baseline comparison.
- No reproducibility controls (unpinned dependencies, missing random seeds).

---

## Final Interpretation

This repository pattern is suitable for the food-delivery reliability problem because it supports:
- End-to-end delay analysis from timestamps to actionable risk signals.
- Identification of high-risk zones and peak delay windows through engineered features.
- A reproducible evaluation process that can be audited and improved.

Most importantly, it can evolve beyond a one-time notebook into a maintainable ML system.

---

## 2-Minute Video Guide

Use this sequence while recording:

1. Start with the business problem and why delay reliability matters.
2. Walk through repository structure and role of each folder.
3. Trace workflow Data -> Features -> Model -> Evaluation using delivery-delay examples.
4. Explain one strength (modular pipeline).
5. Explain one weakness (temporal leakage risk) and how to fix it.

This demonstrates conceptual understanding instead of code recitation.

---

