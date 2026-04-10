"""
Lesson 5.24: Evaluating Regression Models Using MSE and R2

MSE and R2 answer different evaluation questions:
- MSE: How large are prediction errors (squared-loss perspective)?
- R2: How much variance is explained relative to the mean baseline?

=======================================================================
1) MSE FUNDAMENTALS
=======================================================================

Mean Squared Error is the average of squared prediction errors.

Properties:
- errors cannot cancel (all become positive after squaring)
- large errors are penalized disproportionately
- units are squared target units (often hard to communicate directly)

Practical reporting often includes RMSE to restore original target units.

=======================================================================
2) R2 FUNDAMENTALS
=======================================================================

R2 compares your model to always predicting the mean.

Interpretation:
- 1.0: perfect predictions
- 0.0: equal to mean baseline
- < 0.0: worse than mean baseline

R2 is a relative metric, not an absolute error magnitude metric.

=======================================================================
3) WHY BOTH ARE NEEDED
=======================================================================

MSE without R2 lacks baseline context.
R2 without MSE lacks absolute error scale context.

Report both together, with baseline values, for complete interpretation.

=======================================================================
4) BASELINE COMPARISON
=======================================================================

Use DummyRegressor(strategy="mean") baseline.

For a useful model:
- MSE and RMSE should decrease vs baseline
- R2 should increase vs baseline

If R2 is negative on test data, investigate immediately.

=======================================================================
5) CROSS-VALIDATION STABILITY
=======================================================================

Evaluate fold variability, not only means.

Report:
- CV R2 mean +/- std
- CV RMSE mean +/- std

High variance across folds indicates instability and possible overfitting or
dataset subgroup sensitivity.

=======================================================================
6) COMMON PITFALLS
=======================================================================

- reporting only one metric
- skipping baseline comparison
- interpreting R2 as "accuracy"
- comparing MSE across unrelated datasets with different target scales
- forgetting sign flip for neg_mean_squared_error in cross-validation

=======================================================================
7) PRACTICAL REPORT TEMPLATE
=======================================================================

Include:
- test MSE, RMSE, R2 for baseline and model
- deltas (MSE reduction, RMSE reduction, R2 gain)
- CV R2 and RMSE mean +/- std
- note on any negative R2 in folds

=======================================================================
KEY TAKEAWAY
=======================================================================

MSE measures error magnitude with strong penalty on big mistakes.
R2 measures relative explanatory power over a baseline.
Use both, always with baseline and cross-validation context.
"""


if __name__ == "__main__":
    print(__doc__)
