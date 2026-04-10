# Lesson 5.22 Assignment: Training a Linear Regression Model

## Objective
Train a leakage-safe Linear Regression model, compare it against a baseline, validate consistency with cross-validation, and interpret coefficients.

## Part A: Build a Regression Target for This Project

Because the current project target is classification (`is_delayed`), create a continuous target for this assignment using project features.

Example:

```python
y = (
    12
    + 1.35 * df["distance_km"]
    + 0.95 * df["items_count"]
    + 0.015 * df["order_value"]
    + 0.25 * df["day_of_month"]
)
```

Add small random noise for realism.

## Part B: Baseline vs Linear Regression

1. Split train/test first.
2. Train baseline: DummyRegressor(strategy="mean").
3. Train model: Pipeline(StandardScaler + LinearRegression).
4. Evaluate both on test set with MSE, RMSE, MAE, R2.
5. Report metric improvements over baseline.

Starter command:

```bash
python scripts/linear_regression_demo.py
```

Record:
- Baseline RMSE: [write here]
- Baseline MAE: [write here]
- Baseline R2: [write here]
- Model RMSE: [write here]
- Model MAE: [write here]
- Model R2: [write here]
- RMSE improvement: [write here]
- MAE improvement: [write here]
- R2 improvement: [write here]

## Part C: Cross-Validation Stability Check

Use 5-fold CV on training data (`scoring="r2"`).

Record:
- Fold R2 scores: [write here]
- Mean R2: [write here]
- Std R2: [write here]

Interpretation prompt:
- Are fold scores consistent enough to trust model stability?

## Part D: Coefficient Interpretation

Extract and print coefficients from the trained pipeline.

Tasks:
1. Identify top 3 positive coefficients.
2. Identify any negative coefficients.
3. Check if signs align with domain intuition.

Record:
- Intercept: [write here]
- Top positive features: [write here]
- Negative features: [write here]
- Does interpretation make sense? [write here]

## Part E: Leakage and Assumptions Check

Mark TRUE or FALSE and explain:

1. It is safe to fit StandardScaler on the full dataset before splitting.
2. R2 < 0 means model is worse than predicting the mean.
3. Coefficient magnitudes are directly comparable without scaling.
4. Highly correlated features can destabilize coefficient interpretation.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]

## Part F: Verification Checklist

- [ ] Train/test split performed before fitting
- [ ] Baseline and model evaluated on same split and metrics
- [ ] Pipeline used for scaling + model (no leakage)
- [ ] Linear regression beats baseline meaningfully
- [ ] CV results reported
- [ ] Coefficients inspected and interpreted
- [ ] Tests pass

Run:

```bash
python scripts/linear_regression_demo.py
python -m pytest tests/test_regression.py -v
python -m pytest tests/ -v
```

## Reflection

1. Why is baseline comparison mandatory before reporting model performance?
2. When would you move from Linear Regression to Ridge/Lasso or non-linear models?
3. Which metric (RMSE, MAE, R2) is most meaningful for business communication in your setting, and why?

## Suggested Commit Message

lesson-5.22: add linear regression workflow, baseline comparison, and evaluation assignment
