"""
Lesson 5.32: Training a Decision Tree Model

Decision trees learn by repeatedly asking binary questions of the form:
    Is feature X <= threshold Y?

The result is a set of rectangular regions with predictions attached to each
leaf. Unlike linear models, trees do not assume smooth relationships. Unlike
distance-based models, they do not measure similarity with a metric.

=======================================================================
1) WHAT A DECISION TREE IS DOING
=======================================================================

A decision tree builds a model by recursively splitting the training set into
smaller and smaller groups.

At each node it searches over candidate splits and picks the one that improves
purity the most.

For classification:
- purity means one class dominates the node
- the model chooses splits that separate classes cleanly

For regression:
- purity means target values in the node have low variance
- the model chooses splits that reduce variance in the children

The tree keeps splitting until a stopping rule says to stop.

=======================================================================
2) HOW SPLITS ARE CHOSEN
=======================================================================

The algorithm is greedy.

At each node it checks many possible feature-threshold pairs and chooses the
best one right now. It does not look ahead to future splits.

That makes trees fast to fit, but also sensitive to local choices.

Classification split example:
    Is Tenure <= 12 months?
    Is MonthlyCharges > 80?
    Is ContractType = Month-to-month?

Regression split example:
    Split on distance, order_value, or other numeric features to reduce
    target variance inside each child node.

=======================================================================
3) IMPURITY MEASURES FOR CLASSIFICATION
=======================================================================

The split objective is to reduce impurity.

Gini impurity:
    Gini = 1 - sum(p_k^2)

Where p_k is the fraction of class k in the node.

Interpretation:
- Gini = 0 means the node is pure
- higher Gini means the node is more mixed

Entropy:
    Entropy = -sum(p_k * log2(p_k))

Information gain is the entropy reduction after a split.

In practice, Gini and entropy usually produce similar trees. Gini is slightly
faster and is the default in scikit-learn.

=======================================================================
4) IMPURITY FOR REGRESSION
=======================================================================

Regression trees use target variance or squared error reduction.

Each leaf predicts the mean target value of the samples that reach it.

That means regression trees produce step-function predictions and cannot
extrapolate beyond the training range in a meaningful way.

=======================================================================
5) GROWTH AND STOPPING RULES
=======================================================================

The tree grows recursively:
1. Start with the full dataset at the root
2. Evaluate candidate splits
3. Choose the best split
4. Send samples left or right
5. Recurse on each child
6. Stop when a stopping criterion is met

Common stopping rules:
- max_depth
- min_samples_split
- min_samples_leaf
- already pure node
- no split improves impurity enough

Without limits, a tree often memorizes the training set.

=======================================================================
6) OVERFITTING AND COMPLEXITY CONTROL
=======================================================================

Decision trees are high-variance learners.

If unconstrained, they can grow until nearly every training sample sits in its
own leaf. Training accuracy may approach 100%, while test accuracy collapses.

Control complexity with:
- max_depth: deeper trees are more flexible
- min_samples_split: larger values prevent tiny nodes from splitting
- min_samples_leaf: larger values make leaves less specific
- max_features: fewer features per split can reduce variance slightly

The central trade-off is simple:
- more flexibility -> lower bias, higher variance
- more constraints -> higher bias, lower variance

=======================================================================
7) CROSS-VALIDATION FOR DEPTH SELECTION
=======================================================================

Do not guess max_depth.
Use cross-validation on the training set.

Example:

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {"max_depth": range(1, 21)}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True,
)
grid.fit(X_train, y_train)

print(grid.best_params_["max_depth"])
print(grid.best_score_)

Plotting train and CV accuracy across depths makes the bias-variance trade-off
visible.

=======================================================================
8) CLASSIFICATION WORKFLOW
=======================================================================

Typical workflow:
- split data into train and test sets
- build a decision tree classifier
- compare test performance against a baseline
- print the train/test gap
- inspect a confusion matrix or classification report
- check cross-validation stability

Example:

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(tree.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

Always compare to a majority-class baseline.

If the tree cannot beat the baseline, check for:
- too much regularization
- weak features
- class imbalance
- data leakage or preprocessing bugs

=======================================================================
9) VISUALIZING THE TREE
=======================================================================

One of the best features of decision trees is interpretability.

plot_tree can show:
- the split condition at each node
- impurity values
- class counts
- sample counts

If the tree is small enough, you can explain the rules to a domain expert in
plain language.

Feature importance provides a compact summary for larger trees.

It measures how much total impurity reduction each feature contributed across
the tree.

Do not treat feature importance as causality. It is a model-usage signal, not a
scientific proof.

=======================================================================
10) REGRESSION TREE LIMITATIONS
=======================================================================

Regression trees are useful for non-linear patterns, but they have limits:
- predictions are piecewise constant
- they do not extrapolate well
- they can be unstable if grown too deep

They work best when the target has local thresholds or abrupt changes.

=======================================================================
11) STRENGTHS AND WEAKNESSES
=======================================================================

Strengths:
- no feature scaling required
- non-linear relationships are handled naturally
- feature interactions are discovered automatically
- rules are readable and explainable
- mixed feature types are easy to handle after encoding

Weaknesses:
- prone to overfitting
- sensitive to small changes in the data
- axis-aligned splits only
- poor extrapolation for regression
- often outperformed by ensembles

=======================================================================
12) PRACTICAL CHECKLIST
=======================================================================

Before you report results:
- train/test split is correct
- train accuracy and test accuracy are both printed
- train/test gap is assessed
- max_depth or min_samples_leaf is tuned by CV
- baseline performance is reported
- feature importances are reviewed
- tree visualization is sanity-checked
- cross-validation mean and std are included

=======================================================================
KEY TAKEAWAY
=======================================================================

Decision trees are easy to interpret and flexible enough to capture non-linear
patterns, but they overfit aggressively unless you control complexity.

Use them carefully, validate depth with cross-validation, compare against a
baseline, and inspect the resulting rules before trusting the model.
"""


if __name__ == "__main__":
    print(__doc__)