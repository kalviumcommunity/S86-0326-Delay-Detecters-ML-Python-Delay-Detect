# Lesson 5.28 Assignment: Evaluating Classification Models Using F1-Score

## Objective
Evaluate a binary classifier using F1-score, compare against baseline, tune threshold on validation data, and analyze micro/macro/weighted F1 behavior.

## Part A: Run the F1 Workflow

Execute:

```bash
python scripts/f1_evaluation_demo.py
```

Record:
- Baseline precision/recall/F1: [write here]
- Model precision/recall/F1 at threshold 0.5: [write here]
- F1 micro/macro/weighted at threshold 0.5: [write here]
- F1 gain over baseline: [write here]

## Part B: Threshold Optimization

From validation-based threshold tuning output, record:
- Best validation threshold: [write here]
- Best validation F1: [write here]
- Tuned-threshold test precision/recall/F1: [write here]

Interpret:
1. Did tuned threshold improve F1 on test?
2. How did precision and recall shift?
3. Is the new trade-off better for high-stakes positive detection?

## Part C: Classification Report Interpretation

Using default-threshold report:
1. Which class has weaker recall?
2. Which class has weaker precision?
3. What operational risk does this create?

## Part D: Cross-Validation Stability

Record:
- CV binary F1 scores + mean/std: [write here]
- CV macro F1 scores + mean/std: [write here]
- CV weighted F1 scores + mean/std: [write here]

Interpret:
- Is model performance stable?
- Which F1 variant is most conservative here and why?

## Part E: TRUE/FALSE Checks

1. F1 is the arithmetic mean of precision and recall.
2. A model can achieve high F1 with very low recall.
3. Threshold tuning should be done on validation, not test.
4. Macro F1 gives each class equal weight.
5. Reporting F1 without precision/recall can hide important behavior.

Answers:
1. [write here]
2. [write here]
3. [write here]
4. [write here]
5. [write here]

## Part F: Verification Checklist

- [ ] Baseline F1 compared with model F1
- [ ] Precision and recall reported with F1
- [ ] F1 averaging method(s) explicitly reported
- [ ] Threshold tuned on validation only
- [ ] Test metrics evaluated once after threshold selection
- [ ] Cross-validation mean/std reported
- [ ] Tests pass

Run validations:

```bash
python -m pytest tests/test_f1_evaluation.py -v
python -m pytest tests/ -v
```

## Reflection

1. In your use case, should precision and recall be weighted equally?
2. If not, would F-beta be more suitable than F1?
3. What failure mode could still be hidden even with a good F1 score?

## Suggested Commit Message

lesson-5.28: add F1-focused classification evaluation workflow and assignment
