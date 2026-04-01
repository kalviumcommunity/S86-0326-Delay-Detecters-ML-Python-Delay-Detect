# Lesson 5.12 Assignment: Separating Data Loading, Training, and Inference Code

## Objective

Design and validate architectural separation of data loading, training, and
inference code. Ensure each layer has a single, non-overlapping responsibility.

## Part A: Identify Layers in Your Project

Your project has these components present. Verify each layer is separate.

### Data Loading Layer

Files: `src/data_preprocessing.py`

Expected functions:
- `load_data()` - reads raw CSV, returns DataFrame
- Validates file existence, schema consistency
- Does NOT clean, split, or transform

Task:
1. Locate `load_data()` function
2. Confirm it only loads and validates
3. Verify it contains no train/test split
4. Verify it contains no fit() or transform() calls

### Training Layer

Files: `main.py`, `src/train.py`, `src/feature_engineering.py`

Expected behavior:
- Calls load_data() to get raw DataFrame
- Splits train/test
- Calls FIT preprocessing pipeline
- Calls model.fit() on training data only
- Saves artifacts via persistence layer

Task:
1. Trace through main.py flow
2. Confirm all fitting happens during training
3. Verify test data is only used for transformation, not fitting
4. Confirm artifacts are saved after training

### Inference Layer

Files: `src/predict.py`

Expected behavior:
- Loads saved pipeline and model artifacts
- NEVER calls fit_transform()
- Calls transform() and predict()
- Returns predictions

Task:
1. Examine predict.py functions
2. Confirm pipeline.transform() (not fit_transform())
3. Confirm no fitting operations
4. Verify artifacts are loaded, not created

## Part B: Trace Data Flow

Document the complete flow:

### Training Flow

```
Raw Data
  ↓
load_data() [Data Loading Layer]
  ↓
train_test_split() [Training Layer]
  ↓
FIT preprocessing [Training Layer]
  ↓
FIT model [Training Layer]
  ↓
Save artifacts [Training Layer]
```

Task: Map each step to actual functions in your project.

### Inference Flow

```
New Data
  ↓
load_data() [Data Loading Layer]
  ↓
Load pipeline & model [Inference Layer]
  ↓
TRANSFORM (no fit) [Inference Layer]
  ↓
PREDICT [Inference Layer]
  ↓
Return results
```

Task: Map each step to actual functions in your project.

## Part C: Verify No Cross-Contamination

Check that boundaries are enforced:

1. Does training.py import anything from predict.py?
   - Should be: NO
   
2. Does predict.py import anything from train.py?
   - Should be: NO
   
3. Does predict.py ever call fit_transform()?
   - Should be: NO
   
4. Does training ever save intermediate non-artifact files?
   - Should be organized and versioned

## Part D: Test Layer Isolation

You should be able to:

1. Run training without invoking inference:
   ```bash
   python main.py
   ```
   Result: Artifacts saved, no prediction happens

2. Run inference without triggering training:
   ```python
   from src.predict import predict, load_artifacts
   predictions = predict(new_data)
   ```
   Result: Predictions generated from saved artifacts

3. Unit-test each layer independently:
   ```python
   # Test data loading alone
   from src.data_preprocessing import load_data
   df = load_data("data/raw/delivery_data.csv")
   assert not df.empty
   ```

## Part E: Document Architectural Decisions

Create a brief document explaining:

1. Why is data loading separated?
   - Answer: ____________________________________________________

2. Why is training kept separate from inference?
   - Answer: ____________________________________________________

3. How does this architecture prevent data leakage?
   - Answer: ____________________________________________________

4. How does this enable production deployment?
   - Answer: ____________________________________________________

## Part F: Checklist

- [ ] Data loading layer loads only, does not transform
- [ ] Training layer fits only on training data
- [ ] Inference layer loads artifacts, transforms only
- [ ] No circular imports between layers
- [ ] Training and inference are independently executable
- [ ] Each layer is unit-testable
- [ ] Artifacts are saved after training, loaded before inference
- [ ] Code reflects the mental model: Production produces, Inference consumes

## Reflection Questions

1. What would go wrong if inference called fit_transform()?
2. Why should preprocessing parameters learned during training be immutable during inference?
3. How does this architecture scale to multiple models?
4. Why is this architecture essential for production ML systems?

## Suggested Commit Message

lesson-5.12: Document and validate separation of data loading, training, and inference layers

## Submission Standard

Your project should demonstrate clear separation where:
- Anyone reading the code can instantly tell which parts train vs which parts predict
- Training and inference could be run independently
- The pathway for data flowing through the system is obvious
