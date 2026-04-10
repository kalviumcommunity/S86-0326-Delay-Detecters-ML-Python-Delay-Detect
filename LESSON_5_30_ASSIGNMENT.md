# Lesson 5.30 Assignment: Training a K-Nearest Neighbours (KNN) Model

## Objective
Train and evaluate KNN for classification and regression, tune K with cross-validation, and compare performance against baselines.

## Part A: Run the KNN Workflow

Execute:

```bash
python scripts/knn_demo.py
```

Record:
- Classification baseline metrics (accuracy, precision, recall, F1): [write here]
- KNN classification metrics at K=5: [write here]
- Accuracy gain vs baseline: [write here]
- Best K from CV: [write here]
- Best CV accuracy: [write here]

## Part B: K Selection Interpretation

From the K-to-CV-accuracy output:

1. Which K values appear to overfit?
2. At what K does validation performance peak?
3. Does performance decline for very large K values (underfitting signal)?

## Part C: Classification Stability

Record:
- CV accuracy scores + mean/std: [write here]
- CV F1 scores + mean/std: [write here]

Interpret:

1. Is KNN classification performance stable across folds?
2. If variance is high, what might this indicate about local structure?

## Part D: Regression Workflow

From the synthetic regression section, record:

- Baseline RMSE and R2: [write here]
- KNN regression RMSE and R2: [write here]
- R2 gain vs baseline: [write here]
- CV RMSE mean/std: [write here]
- CV R2 mean/std: [write here]

Interpret:

1. Did KNN capture non-linear structure better than baseline?
2. Is regression performance consistent across folds?

## Part E: TRUE/FALSE Checks

1. KNN learns coefficients during training like linear regression.
2. KNN predictions are sensitive to feature scaling.
3. Choosing K should be done using cross-validation on training data.
4. KNN prediction cost grows with training set size.
5. In high-dimensional spaces, nearest-neighbor distance quality can degrade.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Features are scaled inside a pipeline
- [ ] K was selected via cross-validation, not guessed
- [ ] Baseline and KNN compared for classification
- [ ] Baseline and KNN compared for regression
- [ ] CV mean/std reported
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_knn_modeling.py -v
python -m pytest tests/ -v
```

## Reflection

1. In your domain, is prediction latency compatible with KNN's instance-based inference cost?
2. Which matters more for your classification use case: F1 or accuracy, and why?
3. What feature engineering step would most likely improve KNN next?

## Suggested Commit Message

lesson-5.30: add KNN classification/regression workflows with CV-based K tuning