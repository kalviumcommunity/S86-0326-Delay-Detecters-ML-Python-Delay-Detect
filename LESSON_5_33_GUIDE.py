"""
Lesson 5.33: Interpreting Feature Importance from Tree-Based Models

After training tree-based models such as Decision Trees and Random Forests,
feature importance helps answer a critical question:
    Which features are actually driving predictions?

This is useful for two goals:
- practical model improvement (feature pruning and better feature engineering)
- analytical insight (understanding what the model relied on)

=======================================================================
1) WHAT FEATURE IMPORTANCE MEANS IN TREES
=======================================================================

Tree splits are chosen to reduce impurity.

For classification:
- Gini or entropy reduction

For regression:
- variance or squared-error reduction

Each split credits its chosen feature with the weighted impurity reduction at
that node. Summing across all splits gives feature importance.

In scikit-learn this is exposed as:
- feature_importances_

This measure is commonly called:
- Mean Decrease in Impurity (MDI)

MDI values are normalized to sum to 1.0. They are relative, not absolute.

=======================================================================
2) WHY SOME FEATURES LOOK IMPORTANT
=======================================================================

High MDI tends to happen when a feature:
- appears near the root
- splits many samples
- produces large impurity reductions

Low MDI tends to happen when a feature:
- is rarely selected
- appears only deep in the tree
- contributes little additional purity gain

Root-level splits matter disproportionately because they affect more data.

=======================================================================
3) EXTRACTING IMPORTANCE IN SCIKIT-LEARN
=======================================================================

Basic example:

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_

Typical workflow:
- build a sorted table
- create a horizontal bar chart
- inspect the top and bottom features

=======================================================================
4) WHAT IMPORTANCE DOES NOT MEAN
=======================================================================

Feature importance is not causality.

It only means a feature improved prediction for this model on this data.
It does not prove the feature causes the outcome.

Importance is also conditional on the full feature set. If features are highly
correlated, one may absorb most importance while the other appears weak.

=======================================================================
5) MDI BIAS: HIGH CARDINALITY
=======================================================================

MDI can overvalue high-cardinality variables.

Why:
- features with many unique values offer many candidate split points
- greedy search is more likely to find apparently strong splits by chance

Result:
- continuous and high-cardinality features may look too important

Treat suspiciously high-ranked high-cardinality features with caution.

=======================================================================
6) CORRELATED FEATURES SPLIT IMPORTANCE
=======================================================================

When features are strongly correlated, trees may choose one and ignore the
other. This can make the ignored feature appear unimportant even when it has
equivalent signal.

Before interpreting low-importance features, check correlations.

Numerical correlation heatmaps are an essential diagnostic.

=======================================================================
7) PERMUTATION IMPORTANCE
=======================================================================

Permutation importance answers a different question:
    How much does model performance drop when this feature is shuffled?

Algorithm:
1. Measure baseline performance on held-out data.
2. Shuffle one feature column.
3. Re-evaluate performance.
4. Importance = baseline - shuffled score.

Advantages over MDI:
- less biased by cardinality
- directly tied to held-out performance
- model-agnostic in general usage

Important rule:
- compute permutation importance on validation/test data, not training data

Negative permutation importance means the feature may be harming
generalization.

=======================================================================
8) MDI VS PERMUTATION: PRACTICAL WORKFLOW
=======================================================================

Recommended flow:
1. Train and validate a stable model.
2. Compute MDI for fast exploratory ranking.
3. Compute permutation importance for decision-grade validation.
4. Compare rankings and investigate disagreements.
5. Check correlations before dropping features.
6. Retrain after candidate feature removal and verify metric impact.

When rankings disagree strongly:
- inspect correlation structure
- inspect cardinality
- inspect possible leakage

=======================================================================
9) COMMON MISTAKES
=======================================================================

- Treating importance as causation
- Dropping features based only on MDI
- Ignoring correlation groups
- Reporting importance from weak models
- Using a single untuned tree for interpretation

=======================================================================
10) CHECKLIST BEFORE REPORTING IMPORTANCE
=======================================================================

- Model has acceptable held-out performance
- Importance extracted from tuned model settings
- Correlation matrix reviewed
- MDI and permutation compared
- Near-zero or negative permutation features reviewed
- Drop decisions validated through retraining
- Domain sanity check completed

=======================================================================
KEY TAKEAWAY
=======================================================================

Tree feature importance is powerful for model debugging and insight, but it
must be interpreted carefully. Use MDI for quick exploration and permutation
importance for reliable decisions.
"""


if __name__ == "__main__":
    print(__doc__)
