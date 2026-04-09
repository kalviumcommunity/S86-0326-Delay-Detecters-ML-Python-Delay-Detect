"""
Lesson 5.20: Normalizing Features Using MinMaxScaler

This lesson explains how and when to normalize features into a bounded range
using MinMaxScaler, how this differs from standardization, and how to avoid
data leakage in training and production workflows.

=======================================================================
1) WHAT NORMALIZATION IS
=======================================================================

Normalization rescales a feature to a bounded interval (commonly [0, 1]).
MinMaxScaler uses training-set min and max values per feature.

Formula:
    x_scaled = (x - x_min) / (x_max - x_min)

Where:
- x is the original value
- x_min and x_max come from training data only

After fitting on training data:
- minimum training value maps to 0
- maximum training value maps to 1
- all other values map proportionally in between

Key point:
- Relative ordering and spacing are preserved.
- Absolute scale is compressed to a bounded interval.

=======================================================================
2) NORMALIZATION VS STANDARDIZATION
=======================================================================

StandardScaler:
- Formula: (x - mean) / std
- Produces centered data around 0 with unit variance
- Output is unbounded

MinMaxScaler:
- Formula: (x - min) / (max - min)
- Preserves feature shape but rescales to bounded range
- Output is bounded for training data

Use MinMaxScaler when bounded inputs are helpful or required.
Use StandardScaler when centering and variance scaling is preferred.

=======================================================================
3) WHEN MINMAXSCALER HELPS
=======================================================================

Typically useful for scale-sensitive models:
- k-Nearest Neighbors
- k-Means
- SVM (especially RBF kernel)
- Neural networks
- Gradient-based optimization workflows

Why:
- Distance calculations become fair across features
- Gradient updates are more stable with consistent input ranges

=======================================================================
4) WHEN SCALING IS LESS IMPORTANT
=======================================================================

Often unnecessary for tree-based models:
- Decision Trees
- Random Forest
- Gradient Boosted Trees

Reason:
- Trees split on thresholds, not Euclidean distance.
- Feature scale usually does not change split ordering.

=======================================================================
5) THE NON-NEGOTIABLE RULE: SPLIT BEFORE FIT
=======================================================================

Wrong (leakage):
    scaler.fit_transform(X)  # full dataset
    train_test_split(...)

Correct:
    X_train, X_test, y_train, y_test = train_test_split(...)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

Why this matters:
- Fitting on full data lets test statistics influence training.
- Metrics become optimistic and non-reproducible in production.

=======================================================================
6) SCALE ONLY NUMERICAL FEATURES
=======================================================================

Do not apply MinMaxScaler directly to raw categorical columns.
Categorical features should be encoded, not numerically normalized.

Professional approach:
- Use ColumnTransformer
- Apply MinMaxScaler to numerical columns
- Apply OneHotEncoder to categorical columns
- Wrap preprocessing + model in one Pipeline

Benefits:
- Leakage prevention by design
- Single artifact for deployment
- Consistent transform order at training and inference

=======================================================================
7) OUTLIER SENSITIVITY
=======================================================================

MinMaxScaler is sensitive to extreme values.
A single outlier can stretch range and compress most points near 0.

Mitigations:
- cap/winsorize outliers
- log-transform skewed features
- switch to RobustScaler or StandardScaler

Always inspect distributions before finalizing scaling strategy.

=======================================================================
8) PRODUCTION CHECKLIST
=======================================================================

Before shipping:
- Split before fitting
- Fit scaler on training data only
- Transform test/new data with same fitted scaler
- Scale only numerical columns
- Save scaler/pipeline artifact
- Never call fit() during inference
- Validate transformed ranges and monitor drift

=======================================================================
KEY TAKEAWAY
=======================================================================

MinMaxScaler is simple but high-impact for scale-sensitive models.
Use it intentionally, guard against leakage, account for outliers,
and package it inside reproducible pipelines for production reliability.
"""


if __name__ == "__main__":
    print(__doc__)
