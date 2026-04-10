# Lesson 5.26 Assignment: Evaluating Classification Models Using Accuracy

## Objective
Evaluate a classification model using accuracy responsibly by combining baseline comparison, balanced accuracy, confusion-matrix analysis, and cross-validation.

## Part A: Run the Accuracy Workflow

Execute:

```bash
python scripts/accuracy_evaluation_demo.py
```

Record:
- Baseline accuracy: [write here]
- Baseline balanced accuracy: [write here]
- Model accuracy: [write here]
- Model balanced accuracy: [write here]
- Accuracy gain over baseline: [write here]

## Part B: Class Imbalance Check

From demo output, record test class distribution.

Interpret:
1. Is this dataset balanced or imbalanced?
2. Is plain accuracy trustworthy as the primary metric here?
3. Does balanced accuracy change your interpretation?

## Part C: Confusion Matrix Interpretation

From the model confusion breakdown (TN, FP, FN, TP), answer:

1. Which error type is more common (FP or FN)?
2. Which class is harder for the model to detect?
3. Is the model useful for minority-class detection?

## Part D: Cross-Validation Stability

Record from CV output:
- CV accuracy fold scores: [write here]
- Mean CV accuracy +/- std: [write here]
- CV balanced accuracy fold scores: [write here]
- Mean CV balanced accuracy +/- std: [write here]

Interpret:
- Is performance stable across folds?
- Are any folds notably weak?

## Part E: TRUE/FALSE Integrity Checks

1. Accuracy alone is sufficient for imbalanced fraud detection.
2. Majority-class baseline is required context for interpreting accuracy.
3. Balanced accuracy gives equal importance to each class recall.
4. High accuracy always implies high minority recall.
5. Confusion matrix inspection is optional if accuracy is high.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Accuracy computed on held-out test data
- [ ] Majority baseline included
- [ ] Balanced accuracy reported
- [ ] Confusion matrix inspected
- [ ] Cross-validation mean/std reported
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_accuracy_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. When is accuracy a good primary metric?
2. Why can balanced accuracy be more honest on imbalanced datasets?
3. What additional metric would you prioritize for this project and why?

## Suggested Commit Message

lesson-5.26: add accuracy-focused classification evaluation workflow and assignment
