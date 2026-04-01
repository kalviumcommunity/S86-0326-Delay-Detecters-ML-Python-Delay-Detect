"""
Lesson 5.12: Separating Data Loading, Training, and Inference Code

This lesson demonstrates architectural discipline: keeping data loading, training,
and inference logic in separate, non-overlapping modules.

=======================================================================
1) WHY SEPARATION MATTERS
=======================================================================

ML projects have two fundamentally different execution modes:

Training Mode:
- Read raw data
- Split train/test
- Fit preprocessing (imputers, scalers, encoders)
- Fit model
- Evaluate on held-out test set
- Save artifacts

Inference Mode:
- Load saved preprocessing + model
- Read new, unseen data
- Apply preprocessing transformations (NO FITTING)
- Generate predictions
- Return results

These modes must be architecturally separate.
If they are mixed:
- Preprocessing accidentally refits at prediction time
- Data leakage occurs
- Production predictions drift from training behavior
- Reproducibility fails
- Debugging becomes impossible

=======================================================================
2) CONCEPTUAL ARCHITECTURE
=======================================================================

Clean ML system separates three responsibilities:

┌─────────────────────────────────────┐
│  Data Loading Layer                 │
│  - Load raw CSV/database            │
│  - Validate schema                  │
│  - Return structured DataFrame      │
│  - NO cleaning, splitting, or fit   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Training Layer                     │
│  - Split train/test                 │
│  - FIT preprocessing pipeline       │
│  - FIT model                        │
│  - Evaluate                         │
│  - Save artifacts                   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Inference Layer                    │
│  - Load saved artifacts             │
│  - TRANSFORM preprocessing (no fit) │
│  - Predict with model               │
│  - Return results                   │
└─────────────────────────────────────┘

Each layer is independent and testable.

=======================================================================
3) DATA LOADING LAYER
=======================================================================

Responsibility: Read raw data. Nothing else.

What it should do:
- Validate file existence
- Load into DataFrame/structured format
- Check for empty data
- Validate schema consistency
- Raise descriptive errors

What it should NOT do:
- Split train/test
- Fit scalers, encoders
- Modify target variables
- Create derived features
- Handle missing values (that's preprocessing)

Example:

    def load_data(filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        return df

This function is reusable everywhere: training, evaluation, visualization.
It does not embed any transformation logic.

Benefit: You know exactly what raw data looks like before any pipeline touches it.

=======================================================================
4) TRAINING LAYER
=======================================================================

Responsibility: Fit everything. Then save artifacts. Nothing else runs inference here.

What it must do:
- Load raw data
- Split train/test (crucial step)
- FIT preprocessing on TRAIN data only
- FIT model on TRAIN data only
- Evaluate on TEST data
- Save pipeline and model

What it must NOT do:
- Ever call predict() or inference code
- Share state with inference
- Use fit_transform() on test data
- Mix experiment logic with inference

Key principle:
FIT ONLY ON TRAINING DATA.

Example:

    def train_model(data_path: str):
        df = load_data(data_path)
        X = df.drop("target", axis=1)
        y = df["target"]
        
        # SPLIT FIRST
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # FIT on train only
        pipeline = build_pipeline()
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # FIT model on train only
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_processed, y_train)
        
        # SAVE artifacts
        joblib.dump(pipeline, "models/pipeline.pkl")
        joblib.dump(model, "models/model.pkl")

Notice: Test set is NEVER used for fitting. Only transformation.

=======================================================================
5) INFERENCE LAYER
=======================================================================

Responsibility: Load artifacts. Apply without fitting. Generate predictions.

What it must do:
- Load saved preprocessing + model
- Validate input schema
- TRANSFORM input (never fit_transform)
- Generate predictions
- Return structured output

What it must NOT do:
- Ever refit preprocessing
- Ever refit model
- Trigger training logic
- Modify saved artifacts
- Embed experiment code

Key principle:
TRANSFORM ONLY. NEVER FIT.

Example:

    def load_artifacts():
        pipeline = joblib.load("models/pipeline.pkl")
        model = joblib.load("models/model.pkl")
        return pipeline, model

    def predict(new_data: pd.DataFrame):
        pipeline, model = load_artifacts()
        
        # TRANSFORM ONLY
        features = pipeline.transform(new_data)
        
        # PREDICT
        predictions = model.predict(features)
        
        return predictions

This code will NEVER refit preprocessing. 
This guarantees consistency between training and inference.

=======================================================================
6) PREVENTING COMMON MISTAKES
=======================================================================

Mistake 1: Refitting preprocessing during inference
    
    # WRONG
    def predict(new_data):
        pipeline = build_pipeline()
        features = pipeline.fit_transform(new_data)  # REFIT! Bad!
        predictions = model.predict(features)
        return predictions
    
    # CORRECT
    def predict(new_data):
        pipeline = joblib.load("models/pipeline.pkl")
        features = pipeline.transform(new_data)  # TRANSFORM ONLY
        predictions = model.predict(features)
        return predictions

Impact: Wrong approach causes silent data leakage. Predictions drift from training behavior.

Mistake 2: Duplicating train/test split logic

    # WRONG
    # Inside train.py
    train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inside predict.py
    train_test_split(new_data, ...)  # Why? We never split during inference!

Impact: Logic duplication creates inconsistency bugs.

Mistake 3: Mixing global state

    # WRONG
    global_pipeline = None
    global_model = None
    
    def train():
        global global_pipeline, global_model
        global_pipeline = ...
        global_model = ...
    
    def predict():
        # Relies on global state set by train()
        features = global_pipeline.transform(...)

Impact: Unpredictable behavior. Inference depends on training execution order.

Mistake 4: Notebook-centric architecture

    # WRONG (all in one notebook)
    # Cell 1: Load data
    df = pd.read_csv("data.csv")
    
    # Cell 2: Train
    X = df.drop("target", axis=1)
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Cell 3: Predict
    new_data = pd.read_csv("new.csv")
    predictions = model.predict(new_data)

Impact: Not reproducible, not reusable, not testable.

=======================================================================
7) FILE STRUCTURE EXAMPLE
=======================================================================

project/
├── src/
│   ├── data_loader.py       # Load raw data only
│   ├── preprocessing.py      # Build preprocessing pipeline
│   ├── train.py             # Training orchestration
│   ├── evaluate.py          # Evaluation logic
│   ├── predict.py           # Inference only
│   └── persistence.py       # Save/load artifacts
├── data/
│   └── raw/
├── models/
├── requirements.txt
└── README.md

Each file has ONE purpose.
Each layer imports from the layer below.
Layers never import from layers above (no circular dependency).

=======================================================================
8) APPLIED TO THIS PROJECT
=======================================================================

This repository demonstrates 5.12 principles:

Data Loading:
- data_preprocessing.py: load_data() function

Training:
- main.py: orchestrator calling train workflow
- train.py: model.fit() on training data only
- feature_engineering.py: build_preprocessing_pipeline()

Inference:
- predict.py: load_artifacts() + transform() logic
- Never refits preprocessing

The separation is architecturally enforced by module boundaries.

=======================================================================
9) WHY THIS MATTERS IN REAL SYSTEMS
=======================================================================

In production:
- Training runs once per week
- Inference runs thousands of times per minute

If inference triggers training refitting:
- System fails under load
- Predictions become inconsistent
- Monitoring becomes impossible

If preprocessing logic is duplicated:
- Training and inference diverge silently
- Metrics reported in training vs actual production metrics differ

If data loading has hidden transformations:
- Raw data schema is unknown
- Debugging becomes impossible

Separation prevents all these failures.

=======================================================================
10) MENTAL MODEL
=======================================================================

Remember this:

Training produces artifacts (pipeline, model).
Inference consumes artifacts.

During training: FIT everything.
During inference: TRANSFORM only.

Data → [Load] → Raw
Raw → [Fit in Train] → Pipeline & Model
New Data → [Transform using Pipeline] → Predictions

These flows never overlap.

Once you internalize this, you stop writing notebook code.
You start building systems.

=======================================================================
KEY TAKEAWAY
=======================================================================

Separation of data loading, training, and inference is not optional.
It is how you build systems that behave correctly in production.

It prevents accidental refitting.
It prevents data leakage.
It enables independent testing.
It makes failures traceable.

Enforce this separation architecturally.
Your future self will thank you.
"""

if __name__ == "__main__":
    print(__doc__)
