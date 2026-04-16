# Lesson 5.32 Assignment: Training a Decision Tree Model

## Objective
Train and interpret a decision tree model, control overfitting with depth and leaf constraints, and compare against a simple baseline.

## Part A: Run the Demo

Execute:

```bash
python scripts/decision_tree_demo.py
```

Record:
- Baseline accuracy: [write here]
- Tree train accuracy: [write here]
- Tree test accuracy: [write here]
- Train/test gap: [write here]
- Best depth from CV: [write here]
- Best CV accuracy: [write here]

## Part B: Classification Interpretation

Using the printed classification results, answer:
1. Did the tree beat the baseline clearly?
2. Does the train/test gap suggest overfitting?
3. Which metric besides accuracy is worth checking for this target?

## Part C: Depth Tuning

From the depth curve output, record:
- Depth values tested: [write here]
- Mean train accuracy by depth: [write here]
- Mean CV accuracy by depth: [write here]

Answer:
1. Which depth appears to overfit?
2. Which depth looks like the best balance between bias and variance?
3. What would happen if max_depth were removed entirely?

## Part D: Feature Importance

Record the top 5 features from the printed importance table.

Answer:
1. Which features look most influential?
2. Do the top features make sense from a domain perspective?
3. Could any of them be proxy features that deserve caution?

## Part E: Regression Tree Check

From the synthetic regression section, record:
- Tree RMSE: [write here]
- Tree R2: [write here]
- Baseline R2: [write here]
- Train R2: [write here]
- Train/test gap: [write here]

Answer:
1. Does the tree improve over the mean baseline?
2. Is the gap consistent with overfitting or a healthy fit?
3. Why are regression trees poor extrapolators?

## Part F: TRUE/FALSE Checks

1. Decision trees need feature scaling before training.
2. A deeper tree usually has lower bias and higher variance.
3. Gini impurity is a classification impurity measure.
4. Regression trees predict the mean target value in each leaf.
5. A single tree is usually the final best model for tabular data.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part G: Verification Checklist

- [ ] Demo script executed successfully
- [ ] Baseline comparison included
- [ ] Train/test gap reported
- [ ] Depth tuned with cross-validation
- [ ] Tree visualization saved
- [ ] Feature importance reviewed
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_decision_tree_modeling.py -v
python -m pytest tests/ -v
```

## Reflection

1. In your current pipeline, would a tree help more as a standalone model or as a feature for an ensemble?
2. Which constraint, max_depth or min_samples_leaf, is more actionable for your data?
3. What would make a tree easier to trust in front of a non-technical stakeholder?

## Suggested Commit Message

lesson-5.32: add decision tree lesson, demo, and evaluation helpers