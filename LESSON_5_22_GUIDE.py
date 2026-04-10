"""
Lesson 5.22: Training a Linear Regression Model

Linear Regression is typically the first feature-learning model used after a
regression baseline. It predicts a continuous target from a linear combination
of input features.

=======================================================================
1) CORE IDEA
=======================================================================

Model form:
    y_hat = b0 + b1*x1 + b2*x2 + ... + bn*xn

Where:
- y_hat: predicted target
- b0: intercept
- bi: feature coefficients
- xi: feature values

The model learns coefficients that minimize mean squared error on training data.

=======================================================================
2) WHY IT MATTERS
=======================================================================

Linear Regression provides:
- a strong first non-trivial regression model
- coefficient-level interpretability
- a fast, deterministic training process
- a clear benchmark before trying more complex models

Always compare it to a mean/median baseline.

=======================================================================
3) TRAINING PROCESS
=======================================================================

Standard workflow:
1. split data into train/test
2. train baseline (DummyRegressor)
3. train linear regression (prefer Pipeline)
4. evaluate both on same test set
5. compare RMSE, MAE, and R2
6. validate with cross-validation
7. inspect coefficients

=======================================================================
4) METRICS TO REPORT
=======================================================================

- MSE: penalizes larger errors strongly
- RMSE: same unit as target, easier to explain
- MAE: robust absolute error summary
- R2: variance explained relative to mean baseline

Interpretation:
- baseline R2 should be near 0 on test data
- linear model should improve error metrics and R2

=======================================================================
5) SCALING AND PIPELINES
=======================================================================

LinearRegression can fit without scaling, but scaling is useful for:
- coefficient comparability across features
- numerical stability
- consistency with regularized variants (Ridge/Lasso)

Use Pipeline to avoid leakage:
- scaler fits on training data only
- same transform applied to test/inference

=======================================================================
6) COEFFICIENT INTERPRETATION
=======================================================================

Each coefficient indicates expected target change for a one-unit increase in
that feature, holding all others constant.

Cautions:
- high feature correlation makes coefficients unstable
- coefficient magnitudes are not directly comparable unless features are scaled
- prediction quality can be good even when coefficient interpretation is weak

=======================================================================
7) COMMON FAILURE MODES
=======================================================================

- non-linear relationships -> underfitting
- outliers -> distorted coefficients
- multicollinearity -> unstable interpretation
- leakage from preprocessing before split
- reporting model metrics without baseline context

=======================================================================
8) NEXT STEPS WHEN NEEDED
=======================================================================

If linear regression underperforms:
- engineer non-linear features
- try PolynomialFeatures
- use Ridge or Lasso
- evaluate tree-based regressors for complex interactions

=======================================================================
KEY TAKEAWAY
=======================================================================

Linear Regression is simple, fast, and interpretable. Use it after baseline,
compare honestly, and treat it as a foundational regression reference model.
"""


if __name__ == "__main__":
    print(__doc__)
