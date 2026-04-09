"""
Lesson 5.21: Creating a Baseline Model Using Simple Heuristics

A baseline model is the minimum performance bar your trained model must beat.
It is intentionally simple and serves as an honesty check for model quality.

=======================================================================
1) WHY BASELINES ARE REQUIRED
=======================================================================

Without a baseline, model metrics have no context.
A reported 90% accuracy may be useless if a trivial predictor gets 88%.

Baselines help you:
- define a performance floor
- expose class imbalance illusions
- detect suspicious leakage patterns
- justify model complexity with measurable gains

=======================================================================
2) COMMON CLASSIFICATION BASELINES
=======================================================================

1. Majority-class baseline
   Predict the most frequent class for every sample.

2. Stratified random baseline
   Random predictions following training class proportions.

3. Uniform random baseline
   Random predictions with equal class probability.

4. Rule-based heuristic baseline
   Hand-crafted business rule from training data only.

Use sklearn DummyClassifier for consistency and leakage-safe fitting.

=======================================================================
3) COMMON REGRESSION BASELINES
=======================================================================

1. Mean baseline
   Predict training target mean for every sample.

2. Median baseline
   Predict training target median for every sample.

Use sklearn DummyRegressor for these baselines.

=======================================================================
4) METRIC DISCIPLINE
=======================================================================

Always compare baseline and trained model on identical metrics and split.

Classification:
- accuracy
- precision/recall/f1 (especially with imbalance)
- ROC-AUC (if probabilistic outputs exist)

Regression:
- MAE
- MSE/RMSE
- R2

Do not compare model F1 to baseline accuracy. Metric mismatch invalidates
comparison.

=======================================================================
5) LEAKAGE GUARDRAILS
=======================================================================

- Split data before fitting any baseline or model.
- Fit baseline on training data only.
- Derive heuristic thresholds from training data only.
- Evaluate only on held-out test/validation data.

If a baseline score is surprisingly high, investigate leakage immediately.

=======================================================================
6) HOW TO INTERPRET BASELINE RESULTS
=======================================================================

Good model result format:
"Baseline accuracy: 0.62, model accuracy: 0.78, improvement: +0.16"

Also inspect minority-class recall/F1 and confusion matrices.
Small aggregate improvements can still be meaningful if minority recall grows.

=======================================================================
7) INTEGRATING BASELINES INTO PIPELINES
=======================================================================

Baselines should be part of every experiment run:
1) split data
2) run and log baseline metrics
3) run and log model metrics
4) compute explicit improvement over baseline
5) gate promotion on meaningful lift

Treat baseline metrics as first-class outputs in experiment tracking.

=======================================================================
KEY TAKEAWAY
=======================================================================

Baselines are not optional. They anchor reality.
A model is only useful if it clearly and consistently beats a simple baseline.
"""


if __name__ == "__main__":
    print(__doc__)
