"""Lesson 5.34: Improving Model Performance Using GridSearchCV.

After a model works, the next question is: can we do better?
Hyperparameters control model behavior but are not learned from data.
GridSearchCV turns tuning from guesswork into a reproducible search process.

=======================================================================
1) WHY HYPERPARAMETERS MATTER
=======================================================================

Hyperparameters are practitioner-chosen settings fixed before fit.
Examples:
- K in KNN
- max_depth in Decision Trees
- C in Logistic Regression
- n_estimators in Random Forest

Bad choices are predictable:
- too restrictive -> underfitting (high bias)
- too flexible -> overfitting (high variance)

Tuning can produce meaningful improvements in the target metric,
often enough to change deployment decisions.

=======================================================================
2) HOW GRIDSEARCHCV WORKS
=======================================================================

GridSearchCV does five things:
1. Define a hyperparameter grid.
2. Generate all value combinations (Cartesian product).
3. Cross-validate each combination.
4. Select the best mean validation score.
5. Refit best config on all training data.

Step 5 is critical: final model is retrained with full training data,
not one CV fold.

=======================================================================
3) LEAKAGE-SAFE IMPLEMENTATION PATTERN
=======================================================================

Always tune with preprocessing inside a Pipeline.

Why:
- If scaling is fit outside the pipeline, fold-level leakage occurs.
- Pipeline ensures scaling/encoding are fit per training fold only.

Pattern:
- split train/test first
- run GridSearchCV on training set only
- evaluate test set once, after tuning

=======================================================================
4) INTERPRETING SEARCH OUTPUT
=======================================================================

Key attributes:
- search.best_params_
- search.best_score_
- search.best_estimator_
- search.cv_results_

Inspect more than the top row:
- std_test_score: stability across folds
- near-best simpler configs: may generalize better
- edge-of-grid wins: expand the search range

=======================================================================
5) THE SCORING METRIC DRIVES EVERYTHING
=======================================================================

GridSearchCV optimizes exactly the metric in scoring=.
If scoring is wrong, optimization target is wrong.

Recommended defaults:
- balanced classification: accuracy
- imbalanced classification: f1 or roc_auc
- high FN cost: recall
- high FP cost: precision
- regression: neg_mean_squared_error or r2

Remember: neg_* regression metrics are reported negated by sklearn.
Negate back for human reporting.

=======================================================================
6) COMPUTATIONAL TRADE-OFFS
=======================================================================

Total model fits:
- number_of_combinations * CV_folds

Manage cost with:
- n_jobs=-1
- narrower, meaningful grids
- CV=3 for exploration, CV=5/10 for final pass
- sequential tuning of impactful parameters first
- RandomizedSearchCV for large spaces

=======================================================================
7) RANDOMIZEDSEARCHCV
=======================================================================

RandomizedSearchCV samples a fixed number of configurations from
distributions rather than enumerating all combinations.

Use it when:
- 4+ hyperparameters
- wide/continuous ranges
- strict compute budget

In many practical problems, 50-200 random samples reach near-grid quality
at a fraction of the cost.

=======================================================================
8) COARSE-TO-FINE STRATEGY
=======================================================================

Step 1 (coarse): broad sparse grid to locate promising region.
Step 2 (fine): dense search near the promising region.
Step 3: final test evaluation once.

This often reduces compute by 5-10x with similar final quality.

=======================================================================
9) COMMON MISTAKES
=======================================================================

- tuning on test set
- preprocessing outside pipeline
- optimizing irrelevant metric
- reporting only best_score without std
- too-narrow or too-wide grid without rationale

=======================================================================
10) REPORTING CHECKLIST
=======================================================================

A defensible report includes:
- baseline performance
- untuned performance
- tuned CV mean +/- std
- final test score (single evaluation)
- best hyperparameters
- model and metric rationale

=======================================================================
KEY TAKEAWAY
=======================================================================

GridSearchCV is not just an API call; it enforces disciplined model
selection. Combined with leakage-safe pipelines and metric-aligned scoring,
it produces reproducible, defensible improvements.
"""


if __name__ == "__main__":
    print(__doc__)
