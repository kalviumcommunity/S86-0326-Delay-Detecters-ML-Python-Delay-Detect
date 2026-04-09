# Lesson 5.21 Assignment: Creating Baseline Models with Simple Heuristics

## Objective
Establish trustworthy baseline metrics for your delivery-delay project and verify that trained models provide meaningful lift.

## Part A: Majority-Class Baseline (Classification)

1. Load and split project data using existing utilities.
2. Train a baseline with DummyClassifier(strategy="most_frequent").
3. Evaluate on the test set using the same metrics as your model.

Starter:

```python
from src.baselines import train_classification_baseline, evaluate_classification_baseline
from src.data_preprocessing import load_data, clean_data, split_data
from src.config import Config


df = load_data(Config.RAW_DATA_PATH)
df = clean_data(df, target_column=Config.TARGET_COLUMN)
X_train, X_test, y_train, y_test = split_data(df, Config.TARGET_COLUMN)

baseline = train_classification_baseline(X_train, y_train, strategy="most_frequent")
metrics = evaluate_classification_baseline(baseline, X_test, y_test)
print(metrics)
```

Record:
- Baseline accuracy: [write here]
- Baseline weighted F1: [write here]
- Baseline ROC-AUC: [write here]

## Part B: Heuristic Baseline

Implement a simple rule from training data only.

Example rule:
- predict delay (1) if distance_km >= training median distance
- else predict on-time (0)

Tasks:
1. Compute threshold on X_train only.
2. Predict on X_test.
3. Report accuracy and weighted F1.
4. Compare against majority baseline.

Record:
- Heuristic threshold used: [write here]
- Heuristic accuracy: [write here]
- Heuristic weighted F1: [write here]

## Part C: Trained Model vs Baselines

1. Train LogisticRegression (or your current model) on prepared features.
2. Evaluate on the same test split.
3. Compute metric-wise improvement vs majority baseline.

Use this pattern:

```python
from src.baselines import compare_to_baseline

improvement = compare_to_baseline(model_metrics, baseline_metrics)
print(improvement)
```

Record:
- Model accuracy: [write here]
- Model weighted F1: [write here]
- Accuracy lift over baseline: [write here]
- F1 lift over baseline: [write here]

## Part D: Regression Baseline (Conceptual + Optional Coding)

Explain how you would baseline a regression target:
1. mean baseline (DummyRegressor(strategy="mean"))
2. median baseline (DummyRegressor(strategy="median"))
3. evaluation with MAE/RMSE/R2

Optional coding task:
- Use a synthetic continuous target and run both baselines.

## Part E: Leakage and Evaluation Integrity Checks

For each statement, mark TRUE or FALSE and explain.

1. Fitting DummyClassifier on full data before train-test split is acceptable.
2. A model should be compared to baseline using different metrics if needed.
3. Heuristic thresholds should be derived from training data only.
4. High accuracy with near-zero minority recall can still indicate model failure.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]

## Part F: Practical Workflow Check

Run:

```bash
python scripts/baseline_demo.py
python -m pytest tests/ -v
```

Checklist:
- [ ] Baseline metrics computed on held-out test set
- [ ] Heuristic derived from training data only
- [ ] Trained model compared against baseline on same metrics
- [ ] Improvement over baseline reported explicitly
- [ ] No leakage detected
- [ ] Tests pass

## Reflection

1. Why is "model accuracy" without baseline context potentially misleading?
2. If your model barely beats baseline, what should you improve first?
3. In your project, which metric best captures business value and why?

## Suggested Commit Message

lesson-5.21: add baseline modeling utilities, demo workflow, and assignment materials
