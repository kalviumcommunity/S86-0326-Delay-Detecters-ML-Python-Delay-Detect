# Lesson 5.23 Assignment: Evaluating Regression Models Using MAE

## Objective
Evaluate a regression model using MAE, compare it to a baseline, interpret MAE in business terms, and validate reliability with cross-validation.

## Part A: Run the MAE Workflow

Execute:

```bash
python scripts/mae_evaluation_demo.py
```

Record outputs:
- Baseline MAE: [write here]
- Model MAE: [write here]
- MAE improvement: [write here]
- MAE improvement (%): [write here]
- Model MAE as % of mean target: [write here]

## Part B: Compare MAE, RMSE, and R2

From the same run, record:
- Baseline RMSE / R2: [write here]
- Model RMSE / R2: [write here]

Short analysis:
1. Does MAE improvement agree with RMSE and R2 directionally?
2. If RMSE improves much less than MAE, what might that indicate?

## Part C: Cross-Validation with MAE

Use 5-fold CV scores from the demo.

Record:
- CV MAE fold scores: [write here]
- Mean CV MAE: [write here]
- Std CV MAE: [write here]

Interpret:
- Is the MAE stable across folds?
- Would you trust this model in production based on CV spread?

## Part D: Residual Bias Check

Use residual diagnostics from the demo.

Record:
- residual_mean: [write here]
- residual_median: [write here]

Interpret:
- Is the model systematically overpredicting or underpredicting?

## Part E: Baseline and Metric Integrity Checks

Mark TRUE or FALSE with one-line justification.

1. MAE should be computed on training data for final model reporting.
2. MAE and RMSE can be compared directly only because they are in target units.
3. A lower MAE than baseline indicates the model has learned useful signal.
4. In scikit-learn CV, neg_mean_absolute_error should be negated before reporting.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]

## Part F: Verification Checklist

- [ ] MAE computed on held-out test data
- [ ] Baseline MAE and model MAE compared directly
- [ ] MAE interpreted in target units and % of mean target
- [ ] CV MAE mean and std reported
- [ ] Residual bias reviewed
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_mae_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. Why is MAE often easier for business stakeholders to trust than MSE?
2. When would RMSE be preferable as the primary metric?
3. What additional diagnostics would you include before shipping this model?

## Suggested Commit Message

lesson-5.23: add MAE-focused regression evaluation workflow, demo, and assignment
