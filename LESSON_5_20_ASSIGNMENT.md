# Lesson 5.20 Assignment: Normalizing Features with MinMaxScaler

## Objective
Implement leakage-safe MinMax normalization for numerical features, compare it to current scaling behavior, and validate production-ready persistence.

## Part A: Understand Your Current Pipeline

1. Open src/feature_engineering.py and inspect build_preprocessing_pipeline().
2. Confirm which scaler is used by default.
3. Confirm categorical columns are processed separately.

Answer:
- Default scaler: [write here]
- Categorical handling: [write here]

## Part B: Apply MinMaxScaler Correctly

Run this script to see correct MinMax usage end-to-end:

```bash
python scripts/minmax_demo.py
```

The demo should:
- split data before fitting
- fit MinMaxScaler only on training data
- transform test data with the same fitted scaler
- train/evaluate a model
- save pipeline artifact to models/

Record output summary:
- Train min/max per feature: [write here]
- Test range notes: [write here]
- Accuracy/F1: [write here]

## Part C: Leakage Check

For each snippet, mark CORRECT or LEAKY and explain why.

1) 
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

2)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

3)
```python
# inference
X_new_scaled = scaler.fit_transform(X_new)
```

Answers:
- Snippet 1: [write here]
- Snippet 2: [write here]
- Snippet 3: [write here]

## Part D: Outlier Stress Test

In scripts/minmax_demo.py, inspect the outlier example section.

Tasks:
1. Explain how one extreme value changes normalized spacing.
2. Describe one mitigation strategy (cap, log, robust scaling, etc.).
3. State which mitigation you would use for highly skewed monetary features and why.

Answer:
[write here]

## Part E: Project Integration

Implement a MinMax-enabled preprocessing pipeline via:
- build_preprocessing_pipeline(..., scaler_type="minmax")

Then answer:
1. Which model types in this project benefit most from MinMax scaling?
2. Which model types likely do not benefit much?
3. Why should tree-only pipelines often skip scaling?

Answer:
[write here]

## Part F: Verification Checklist

- [ ] MinMaxScaler fit only on training data
- [ ] Test/new data uses transform() only
- [ ] Numerical and categorical transforms are separated
- [ ] Pipeline can be serialized and reused
- [ ] No leakage from preprocessing
- [ ] Unit tests pass

## Suggested Commands

```bash
python scripts/minmax_demo.py
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/ -v
```

## Suggested Commit Message

lesson-5.20: add MinMaxScaler workflow, leakage-safe normalization, and assignment materials

## Reflection

1. Why can MinMaxScaler hurt performance when outliers are extreme?
2. Why is transform-only inference mandatory?
3. Why is ColumnTransformer preferred over manual column handling in production?
