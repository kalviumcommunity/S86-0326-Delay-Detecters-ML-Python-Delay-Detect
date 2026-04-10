# Lesson 5.29 Assignment: Creating and Interpreting a Confusion Matrix

## Objective
Generate, interpret, and compare confusion matrices for baseline and trained classifiers, including normalized and threshold-shifted views.

## Part A: Run the Confusion Matrix Workflow

Execute:

```bash
python scripts/confusion_matrix_demo.py
```

Record:
- Baseline confusion matrix cells (TN, FP, FN, TP): [write here]
- Model confusion matrix cells (TN, FP, FN, TP): [write here]
- Baseline derived metrics (accuracy, precision, recall, F1): [write here]
- Model derived metrics (accuracy, precision, recall, F1): [write here]

## Part B: Interpretation

From the model confusion matrix, answer:

1. Is the model making more false positives or false negatives?
2. Which error type is costlier in your domain?
3. Should you optimize for fewer FP or fewer FN next?

## Part C: Normalized Matrix Analysis

Using the row-normalized matrix output (`normalize="true"`):

1. What is recall for class 0?
2. What is recall for class 1?
3. Does this reveal anything that raw counts hide?

## Part D: Threshold Effects

From thresholds `0.3`, `0.5`, and `0.7`, record confusion cells:

- Threshold 0.3: TP [ ], FP [ ], FN [ ], TN [ ]
- Threshold 0.5: TP [ ], FP [ ], FN [ ], TN [ ]
- Threshold 0.7: TP [ ], FP [ ], FN [ ], TN [ ]

Interpret:

1. How does lowering threshold affect FP and FN?
2. Which threshold would you choose for recall-priority use cases?
3. Which threshold would you choose for precision-priority use cases?

## Part E: Multi-Class Interpretation

From the synthetic multi-class matrix in the demo:

1. Identify the top confused class pair.
2. Why is this pair a candidate for feature engineering?
3. Which class appears easiest to separate and why?

## Part F: TRUE/FALSE Checks

1. In scikit-learn, rows are actual labels and columns are predicted labels.
2. A model can have high accuracy and still fail on the minority class.
3. F1 uses TP, FP, FN and does not use TN.
4. Raw confusion matrices are always sufficient for imbalanced datasets.
5. Threshold tuning should be done on validation data, not test data.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part G: Verification Checklist

- [ ] Baseline and model confusion matrices computed on test data
- [ ] Binary cells (TN, FP, FN, TP) extracted correctly
- [ ] Metrics derived from confusion cells match sklearn expectations
- [ ] Normalized matrix inspected
- [ ] Threshold-based confusion shifts reviewed
- [ ] Multi-class off-diagonal confusions interpreted
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_confusion_matrix_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. Which confusion-cell change would create the biggest business impact for your use case?
2. How would you communicate FP/FN trade-offs to a non-technical stakeholder?
3. What monitoring metric in production would you anchor to confusion matrix behavior?

## Suggested Commit Message

lesson-5.29: add confusion matrix evaluation workflow, interpretation tools, and assignment