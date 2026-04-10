"""
Lesson 5.23: Evaluating Regression Models Using MAE

MAE (Mean Absolute Error) measures average prediction error magnitude in the
same units as the target, making it highly interpretable.

=======================================================================
1) WHAT MAE MEASURES
=======================================================================

MAE is the average absolute difference between actual and predicted values.
It treats over- and under-prediction equally and does not square errors.

Implication:
- MAE is robust to occasional outliers compared to RMSE.
- MAE is easy to communicate: "on average, predictions are off by X units".

=======================================================================
2) MAE VS MSE VS RMSE
=======================================================================

- MAE: linear penalty, target units, lower outlier sensitivity
- MSE: quadratic penalty, squared units, high outlier sensitivity
- RMSE: square-rooted MSE, target units, still high outlier sensitivity

Choose MAE when average absolute miss is the key business concern.

=======================================================================
3) INTERPRETATION RULES
=======================================================================

Never report MAE in isolation. Always provide:
1) baseline MAE comparison
2) MAE as % of mean target
3) cross-validation mean and standard deviation

This gives relevance, scale context, and stability evidence.

=======================================================================
4) BASELINE COMPARISON
=======================================================================

For regression, use DummyRegressor(strategy="mean") baseline.
Model usefulness is demonstrated by reducing MAE relative to baseline.

Example narrative:
"Baseline MAE = 12.4, model MAE = 7.9, improvement = 36.3%."

=======================================================================
5) CROSS-VALIDATION WITH MAE
=======================================================================

In scikit-learn, use scoring="neg_mean_absolute_error" and negate values.

Why negative:
- scikit-learn assumes higher score is better for all scorers.
- MAE is a loss (lower is better), so values are sign-flipped.

Report:
- per-fold MAE
- mean CV MAE
- std CV MAE

=======================================================================
6) PITFALLS TO AVOID
=======================================================================

- reporting MAE without baseline
- mixing metrics between baseline and model
- interpreting MAE after log-transform without back-transforming
- skipping residual checks for directional bias
- claiming success from train MAE without held-out/CV validation

=======================================================================
7) PRACTICAL REPORTING TEMPLATE
=======================================================================

Include all of the following:
- Test MAE (baseline and model)
- MAE improvement and percentage improvement
- Model MAE as % of mean target
- RMSE and R2 as complementary context
- CV MAE mean +/- std
- residual mean near zero (bias check)

=======================================================================
KEY TAKEAWAY
=======================================================================

MAE is often the most business-aligned regression metric because it directly
answers: "On average, how wrong are we?" Use it with baseline comparison,
CV stability, and residual checks for honest model evaluation.
"""


if __name__ == "__main__":
    print(__doc__)
