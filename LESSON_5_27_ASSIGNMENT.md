# Lesson 5.27 Assignment: Evaluating Classification Models Using Precision and Recall

## Objective
Evaluate a classifier with precision and recall, analyze trade-offs, tune threshold intentionally, and compare against baseline.

## Part A: Run the Precision/Recall Workflow

Execute:

```bash
python scripts/precision_recall_demo.py
```

Record:
- Baseline precision: [write here]
- Baseline recall: [write here]
- Baseline F1: [write here]
- Model precision (threshold=0.5): [write here]
- Model recall (threshold=0.5): [write here]
- Model F1 (threshold=0.5): [write here]
- Model F2 (threshold=0.5): [write here]

## Part B: Threshold Tuning

Using the demo output for target recall >= 0.80:

Record:
- Selected threshold: [write here]
- Tuned precision: [write here]
- Tuned recall: [write here]
- Tuned F2: [write here]

Interpret:
1. How did precision change after increasing recall?
2. Is this trade-off acceptable for fraud/disease-style use cases?

## Part C: Classification Report Analysis

From the default-threshold report, answer:

1. Which class has weaker recall?
2. Which class has weaker precision?
3. What operational risk does this imply?

## Part D: Cross-Validation Stability

Record:
- CV precision scores + mean/std: [write here]
- CV recall scores + mean/std: [write here]
- CV F1 scores + mean/std: [write here]

Interpret:
- Is performance stable across folds?
- Which metric varies most, and what might that indicate?

## Part E: TRUE/FALSE Checks

1. High precision guarantees high recall.
2. Lowering threshold generally increases recall.
3. Majority baseline can have high accuracy but zero minority recall.
4. F1 is the arithmetic mean of precision and recall.
5. Threshold 0.5 is always optimal for business goals.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Precision/Recall/F1 computed on held-out test data
- [ ] Majority-class baseline compared
- [ ] Threshold trade-off analyzed
- [ ] Classification report reviewed per class
- [ ] Cross-validation mean/std reported
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_precision_recall_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. In your domain, which error is costlier: false positives or false negatives?
2. How would that cost structure drive your threshold choice?
3. Which metric would you monitor in production first and why?

## Suggested Commit Message

lesson-5.27: add precision-recall evaluation workflow, threshold tuning, and assignment
