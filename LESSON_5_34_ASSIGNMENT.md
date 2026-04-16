# Lesson 5.34 Assignment: Improving Model Performance Using GridSearchCV

## Objective
Tune a classification model with GridSearchCV using a leakage-safe pipeline, compare against baseline and untuned model, and report results transparently.

## Part A: Run the Demo

Execute:

```bash
python scripts/gridsearch_tuning_demo.py
```

Record:
- Baseline test accuracy: [write here]
- Untuned test accuracy: [write here]
- Best GridSearch params: [write here]
- Best GridSearch CV F1: [write here]
- Final test score: [write here]

## Part B: Result Interpretation

Using the printed top configurations:

1. Is the best config stable (check std_test_score)?
2. Is there a simpler near-best config with similar score?
3. Is the best n_neighbors at the edge of your grid?

## Part C: Leakage Audit

Answer each with yes/no and one-line evidence:

1. Was train/test split done before tuning?
2. Was scaling inside the pipeline?
3. Was the test set evaluated once at the end?

## Part D: Metric Alignment

1. Why might accuracy be wrong for imbalanced classes?
2. Which metric would you choose if false negatives are costly?
3. Which metric would you choose if false positives are costly?

## Part E: RandomizedSearchCV Comparison

From the demo output:

- Randomized best params: [write here]
- Randomized best CV F1: [write here]

Answer:
1. How close is RandomizedSearchCV to GridSearchCV in your run?
2. Would you prefer randomized search for a larger hyperparameter space? Why?

## Part F: Coarse-to-Fine Design

Propose your own two-stage KNN tuning plan:

- Coarse grid: [write here]
- Fine grid: [write here]

Explain why your fine grid is centered where it is.

## Validation Commands

```bash
python -m pytest tests/test_hyperparameter_tuning.py -v
python -m pytest tests/ -v
```

## Reflection

1. What was the most surprising hyperparameter effect in the search table?
2. Did tuning move your model enough to change deployment confidence?
3. What is your next tuning target model and metric?

## Suggested Commit Message

lesson-5.34: add gridsearchcv tuning lesson, demo, and tests
