# Lesson 5.25 Assignment: Training a Logistic Regression Classification Model

## Objective
Train a leakage-safe Logistic Regression classifier, compare it against a majority baseline, and evaluate with classification metrics and cross-validation.

## Part A: Run End-to-End Workflow

Execute:

```bash
python scripts/logistic_regression_demo.py
```

Record:
- Baseline accuracy: [write here]
- Baseline ROC-AUC: [write here]
- Baseline F1: [write here]
- Logistic accuracy: [write here]
- Logistic ROC-AUC: [write here]
- Logistic F1: [write here]

## Part B: Baseline Improvement

From the demo output, record:
- Accuracy gain: [write here]
- ROC-AUC gain: [write here]
- F1 gain: [write here]

Interpret:
1. Does Logistic Regression clearly beat baseline?
2. If accuracy gain is small but AUC gain is meaningful, what does that imply?

## Part C: Cross-Validation Stability

Record from CV output:
- CV AUC fold scores: [write here]
- Mean CV AUC +/- std: [write here]
- CV F1 fold scores: [write here]
- Mean CV F1 +/- std: [write here]

Interpret:
- Is classifier behavior stable across folds?
- Are there folds indicating weak generalization?

## Part D: Coefficient and Odds-Ratio Interpretation

From coefficient table, identify:
1. Top features by absolute coefficient.
2. One feature with odds_ratio > 1 and its interpretation.
3. One feature with odds_ratio < 1 and its interpretation.

Record:
- Intercept: [write here]
- Top features: [write here]
- Example positive-effect feature: [write here]
- Example negative-effect feature: [write here]

## Part E: Integrity Checks (TRUE/FALSE)

1. Logistic Regression should be evaluated only with accuracy.
2. A majority-class baseline usually has ROC-AUC near 0.5.
3. Stratified splitting is recommended for binary classification.
4. Coefficient magnitudes are directly comparable without scaling.
5. Convergence warnings can be ignored if test accuracy is high.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Train/test split used with stratify
- [ ] Baseline and model evaluated on same split
- [ ] Accuracy, precision, recall, F1, ROC-AUC reported
- [ ] Cross-validation mean and std reported
- [ ] Coefficients and odds ratios interpreted
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_logistic_classification.py -v
python -m pytest tests/ -v
```

## Reflection

1. Why is ROC-AUC often more informative than accuracy for imbalanced classes?
2. When would you use class_weight="balanced"?
3. What would justify moving beyond Logistic Regression to a more complex model?

## Suggested Commit Message

lesson-5.25: add logistic regression classification workflow, baseline comparison, and assignment
