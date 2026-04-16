# Lesson 5.33 Assignment: Interpreting Feature Importance from Tree-Based Models

## Objective
Interpret feature importance from Random Forest models, compare MDI versus permutation importance, and validate feature-removal decisions with retraining.

## Part A: Run the Demo

Execute:

```bash
python scripts/feature_importance_demo.py
```

Record:
- Train accuracy: [write here]
- Test accuracy: [write here]
- Train/test gap: [write here]
- Top 5 MDI features: [write here]
- Top 5 permutation features: [write here]

## Part B: Ranking Comparison

From the rank-disagreement output:

1. Which features have the largest rank delta?
2. Do those features have high cardinality or known correlation partners?
3. Which ranking would you trust more for feature-removal decisions, and why?

## Part C: Correlation Analysis

Using the printed high-correlation pairs and heatmap:

1. List feature pairs with abs correlation >= 0.80.
2. Did any low-importance feature belong to a high-correlation pair?
3. Why can low importance be misleading for correlated features?

## Part D: Drop-Candidate Validation

From the drop-impact section:

- Dropped features: [write here]
- Baseline accuracy: [write here]
- Dropped-model accuracy: [write here]
- Accuracy delta: [write here]

Answer:
1. Was dropping the candidate features beneficial, neutral, or harmful?
2. If harmful, what does that suggest about the original importance estimates?
3. Would you keep or remove those features in production?

## Part E: TRUE/FALSE Checks

1. A high MDI score proves a feature is causal.
2. Permutation importance should be computed on held-out data.
3. Correlated features can split importance unevenly in trees.
4. Negative permutation importance can indicate a noisy or harmful feature.
5. It is safe to drop features using only MDI rankings.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Demo script executed successfully
- [ ] MDI table reviewed
- [ ] Permutation table reviewed
- [ ] Rank disagreements interpreted
- [ ] Correlation structure checked
- [ ] Drop decision validated by retraining
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_feature_importance_analysis.py -v
python -m pytest tests/ -v
```

## Reflection

1. Which feature in your pipeline surprised you most in permutation ranking?
2. Which feature group appears redundant due to correlation?
3. What feature-collection decision would you change based on this analysis?

## Suggested Commit Message

lesson-5.33: add feature-importance analysis workflow with MDI and permutation checks
