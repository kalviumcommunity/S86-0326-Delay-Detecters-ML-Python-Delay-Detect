# Lesson 5.31 Assignment: Understanding Bias and Variance Through Model Behavior

## Objective
Diagnose underfitting and overfitting using train/test metrics, cross-validation stability, and learning-curve patterns, then apply targeted fixes.

## Part A: Run the Bias-Variance Demo

Execute:

```bash
python scripts/bias_variance_demo.py
```

Record for each K value shown:
- Train accuracy: [write here]
- Test accuracy: [write here]
- Train/test gap: [write here]
- Diagnosis: [write here]

## Part B: Identify Failure Modes

Using your recorded values, classify each model as one of:
- High bias
- High variance
- Good fit
- Mixed/unclear

Then justify each diagnosis in 2-3 lines using train score, test score, and gap.

## Part C: Learning Curve Interpretation

From the printed learning-curve summaries in the demo, record:
- Final train mean accuracy: [write here]
- Final validation mean accuracy: [write here]
- Final gap: [write here]
- Interpretation label: [write here]

Answer:
1. Which setting shows persistent high variance?
2. Which setting is closer to high bias?
3. Would adding more data likely help both equally? Why or why not?

## Part D: Cross-Validation Stability

For one model of your choice, compute 5-fold CV accuracy and report:
- Scores list: [write here]
- Mean: [write here]
- Std: [write here]

Interpret:
1. Is the model stable across folds?
2. What does high std imply even if mean accuracy is acceptable?

## Part E: Intervention Plan

Choose one high-bias case and one high-variance case from your run.

For each case, propose exactly 3 interventions.

High-bias interventions (example directions):
- increase complexity
- reduce regularization
- add interaction/non-linear features

High-variance interventions (example directions):
- increase regularization
- simplify model
- add data

## Part F: TRUE/FALSE Checks

1. A small train/test gap always means the model is good.
2. More data is usually most effective for high-variance models.
3. High training accuracy with low test accuracy is a variance signal.
4. Increasing K in KNN generally increases bias and decreases variance.
5. Bias-variance trade-off can be fully eliminated with enough tuning.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part G: Verification Checklist

- [ ] Demo script executed successfully
- [ ] Train/test behavior diagnosed for each model
- [ ] Learning-curve outputs interpreted
- [ ] CV mean and std reported
- [ ] At least one bias fix and one variance fix proposed
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_bias_variance.py -v
python -m pytest tests/ -v
```

## Reflection

1. In your own project, which is currently the bigger risk: underfitting or overfitting?
2. Which single metric pair will you monitor first in future experiments?
3. What model-complexity lever is most actionable for your pipeline right now?

## Suggested Commit Message

lesson-5.31: add bias-variance diagnostics, learning-curve demo, and assignment
