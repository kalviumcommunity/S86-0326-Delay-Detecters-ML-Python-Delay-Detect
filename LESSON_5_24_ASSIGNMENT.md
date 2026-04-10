# Lesson 5.24 Assignment: Evaluating Regression Models Using MSE and R2

## Objective
Evaluate regression models using MSE and R2 together, compare against baseline, and validate stability with cross-validation.

## Part A: Run the MSE/R2 Evaluation Demo

Execute:

```bash
python scripts/mse_r2_evaluation_demo.py
```

Record:
- Baseline MSE: [write here]
- Baseline RMSE: [write here]
- Baseline R2: [write here]
- Model MSE: [write here]
- Model RMSE: [write here]
- Model R2: [write here]

## Part B: Baseline Comparison Interpretation

Use comparison outputs from the demo.

Record:
- MSE reduction: [write here]
- RMSE reduction: [write here]
- R2 gain: [write here]

Interpret:
1. Is the model meaningfully better than baseline?
2. Are both absolute and relative metrics aligned?

## Part C: Cross-Validation Stability

Record from demo:
- CV R2 fold scores: [write here]
- Mean CV R2 +/- std: [write here]
- CV RMSE fold scores: [write here]
- Mean CV RMSE +/- std: [write here]

Interpret:
- Is model performance stable across folds?
- Any folds with suspiciously low or negative R2?

## Part D: Metric Reasoning Check

Mark TRUE or FALSE with one-line explanation.

1. A lower MSE always means better performance, regardless of dataset scale.
2. R2 = 0 means model is equivalent to predicting the mean.
3. Negative test R2 indicates model underperforms the mean baseline.
4. scikit-learn neg_mean_squared_error CV scores should be negated before reporting MSE.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]

## Part E: Reporting Checklist

- [ ] Test-set MSE, RMSE, and R2 computed
- [ ] Baseline and model metrics reported side-by-side
- [ ] Comparison deltas reported
- [ ] Cross-validation means and standard deviations reported
- [ ] No unexplained negative R2 values
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_mse_r2_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. Why does MSE penalize outliers more heavily than MAE?
2. Why should R2 never be reported without baseline context?
3. If MSE improves but R2 barely changes, what might be happening in the data?

## Suggested Commit Message

lesson-5.24: add MSE/R2 evaluation workflow, baseline comparison, and assignment
