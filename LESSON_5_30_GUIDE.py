"""
Lesson 5.30: Training a K-Nearest Neighbours (KNN) Model

KNN predicts from local similarity rather than learned coefficients.
It stores training examples and performs distance-based voting/averaging.

=======================================================================
1) WHAT KNN DOES
=======================================================================

- Classification: majority vote of K nearest neighbors
- Regression: average target value of K nearest neighbors

KNN is instance-based and lazy: training is light, prediction is heavy.

=======================================================================
2) DISTANCE IS THE MODEL
=======================================================================

Common metrics:
- Euclidean (p=2): default for many tabular datasets
- Manhattan (p=1): can be more robust to large per-feature deviations
- Cosine: often useful in high-dimensional sparse text settings

Distance choice changes which samples are neighbors, so it changes all
predictions.

=======================================================================
3) SCALING IS NON-NEGOTIABLE
=======================================================================

KNN is sensitive to raw feature scale. Without scaling, high-range features
dominate distance and drown out lower-range but informative features.

Use a leakage-safe pipeline:
- fit scaler on train only
- apply same fitted scaler to validation/test/inference

=======================================================================
4) CHOOSING K AND BIAS-VARIANCE
=======================================================================

Small K (for example K=1):
- low bias, high variance, overfitting risk

Large K:
- high bias, low variance, underfitting risk

Select K with cross-validation over a candidate range (often odd K values for
binary classification).

=======================================================================
5) BASELINE COMPARISON
=======================================================================

Always compare KNN against simple baselines:
- classification: majority class (DummyClassifier)
- regression: mean prediction (DummyRegressor)

If KNN does not beat baseline clearly, revisit feature quality, scaling, K,
and signal-to-noise ratio.

=======================================================================
6) CURSE OF DIMENSIONALITY
=======================================================================

As feature dimension grows:
- nearest and farthest points become similarly distant
- local neighborhoods lose meaning
- KNN performance often degrades

Possible mitigation:
- feature selection
- dimensionality reduction (for example PCA)

=======================================================================
7) WHEN KNN FITS AND WHEN IT DOES NOT
=======================================================================

KNN can work well when:
- dataset is small to medium
- feature space is relatively low-dimensional
- local structure is meaningful

KNN often struggles when:
- prediction latency must be very low
- dataset is very large
- features are high-dimensional and noisy

=======================================================================
KEY TAKEAWAY
=======================================================================

KNN is simple to describe but demands disciplined preprocessing and validation.
Scale features, tune K by cross-validation, and prove value against baseline
before deployment decisions.
"""


if __name__ == "__main__":
    print(__doc__)