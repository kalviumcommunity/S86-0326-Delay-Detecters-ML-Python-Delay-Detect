# Lesson 5.16 Assignment: Splitting Data into Training and Testing Sets

## Overview

This assignment tests your understanding of proper data splitting strategies, data leakage prevention, and honest model evaluation. You will work with your existing delivery delay prediction project to verify split integrity, analyze in-project implementation, and recognize common leakage mistakes.

---

## Part A: Train-Test Split Fundamentals from Your Project

### Task 1: Verify Your Project's Train-Test Split

Look at the `src/data_preprocessing.py` file in your project. Find the `split_data()` function.

1. **What split ratio does your project use?**

   ```python
   # Look for this in your code
   from src.data_preprocessing import split_data
   ```

   **Your finding:**

   [Record the split ratio]

2. **What is the random_state value used?**

   **Your finding:**

   [Record the value]

3. **For delivery delay prediction, is stratification being used? Should it be?**

   **Your answer with reasoning:**

   [Explain whether stratification is appropriate and why]

4. **Run this code to inspect your actual split:**

   ```python
   import pandas as pd
   from src.data_preprocessing import load_data, split_data
   
   df = load_data('data/raw/delivery_data.csv')
   X, y = df.drop('DeliveryDelayed', axis=1), df['DeliveryDelayed']
   
   X_train, X_test, y_train, y_test = split_data(X, y)
   
   print(f"Training set size: {X_train.shape}")
   print(f"Test set size: {X_test.shape}")
   print(f"Total: {X_train.shape[0] + X_test.shape[0]} vs {X.shape[0]}")
   print(f"\nClass distribution in training set:")
   print(y_train.value_counts(normalize=True))
   print(f"\nClass distribution in test set:")
   print(y_test.value_counts(normalize=True))
   ```

   **Your output:**

   [Paste the output]

5. **Are the class distributions similar between training and test sets? Why or why not?**

   **Your analysis:**

   [Analyze the proportions]

---

## Part B: The Critical Rule - Splitting Before Fitting

### Task 1: Identify Leakage in Incorrect Approaches

For each code snippet below, identify:
1. What is the mistake?
2. What data leakage occurs?
3. What is the correct approach?

#### Scenario 1: Scaling Before Splitting

```python
# Code from a student's project
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on entire dataset

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

1. **What is wrong here?**

   [Your identification]

2. **What data has leaked and how?**

   [Your explanation]

3. **What is the correct code?**

   [Write the correct version]

---

#### Scenario 2: Feature Selection Before Splitting

```python
# Code from another student's project
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)  # Select features from full data

X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

1. **What is wrong?**

   [Your identification]

2. **Why does this invalidate evaluation?**

   [Your explanation]

3. **What is the correct approach?**

   [Write the correct version]

---

#### Scenario 3: Oversampling Before Splitting

```python
# To handle class imbalance
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)  # Oversample entire dataset

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

1. **What is the problem?**

   [Your identification]

2. **How does SMOTE create synthetic samples?**

   [Explain briefly]

3. **Why is this a problem?**

   [Your explanation of leakage]

4. **What is the correct approach?**

   [Write the correct version]

---

### Task 2: Review Your Project's Preprocessing Pipeline

Look at `src/feature_engineering.py` in your project.

1. **Where is preprocessing fitted in your pipeline?**

   Find where `fit_transform()` is called. Is it on training data or all data?

   **Your finding:**

   [Record what you find]

2. **Is there a separation between fit and transform in your code?**

   Look at the `apply_preprocessing_pipeline()` function.

   **Your analysis:**

   [Explain what you see]

3. **Does your project prevent leakage? Explain.**

   **Your assessment:**

   [Evaluate the quality]

---

## Part C: Stratification for Classification

### Task 1: Class Balance Analysis

Your project predicts delivery delays (binary classification).

1. **Load your data and check class distribution:**

   ```python
   import pandas as pd
   
   df = pd.read_csv('data/raw/delivery_data.csv')
   print("Original class distribution:")
   print(df['DeliveryDelayed'].value_counts(normalize=True))
   ```

   **Your output:**

   [Paste results]

2. **Is this dataset balanced or imbalanced?**

   **Your assessment:**

   [Classify and explain]

3. **If split without stratification, what could go wrong?**

   **Your answer:**

   [Explain potential issues]

4. **Verify stratification is working in your split:**

   ```python
   from src.data_preprocessing import load_data, split_data
   
   df = load_data('data/raw/delivery_data.csv')
   X, y = df.drop('DeliveryDelayed', axis=1), df['DeliveryDelayed']
   
   X_train, X_test, y_train, y_test = split_data(X, y)
   
   print("Train class distribution:")
   print(y_train.value_counts(normalize=True))
   print("\nTest class distribution:")
   print(y_test.value_counts(normalize=True))
   print("\nDifference:")
   train_prop_1 = (y_train == 1).sum() / len(y_train)
   test_prop_1 = (y_test == 1).sum() / len(y_test)
   print(f"Train % of class 1: {train_prop_1:.4f}")
   print(f"Test % of class 1: {test_prop_1:.4f}")
   print(f"Difference: {abs(train_prop_1 - test_prop_1):.4f}")
   ```

   **Your output:**

   [Paste results]

5. **Is stratification working as expected? How do you know?**

   **Your analysis:**

   [Evaluate whether proportions are preserved]

---

## Part D: Time-Series vs Random Splitting

### Task 1: When Would You Use Time-Based Splitting?

For each scenario, decide: Time-based split or random stratified split?

1. **Predicting house prices in a suburb based on historical sales.**
   - 2020-2023 sales records
   - Features: location, size, age
   
   Your choice: ________ (Time-based / Random Stratified)
   
   Reasoning:
   
   [Your reasoning]

2. **Medical diagnosis: predicting disease presence from patient tests.**
   - Data: 1000 patient records
   - Features: test results, demographics
   - Target: disease present/absent
   
   Your choice: ________ (Time-based / Random Stratified)
   
   Reasoning:
   
   [Your reasoning]

3. **Stock price prediction using historical daily prices.**
   - Data: 5 years of daily stock prices
   - Features: technical indicators from past 30 days
   - Target: price goes up/down next day
   
   Your choice: ________ (Time-based / Random Stratified)
   
   Reasoning:
   
   [Your reasoning]

4. **Text sentiment classification.**
   - Data: 10,000 movie reviews
   - Features: word frequencies
   - Target: positive/negative sentiment
   
   Your choice: ________ (Time-based / Random Stratified)
   
   Reasoning:
   
   [Your reasoning]

5. **Network intrusion detection.**
   - Data: server logs labeled as attack/normal
   - Features: protocol, bytes, duration
   - Target: attack/normal
   
   Your choice: ________ (Time-based / Random Stratified)
   
   Reasoning:
   
   [Your reasoning]

---

## Part E: Data Leakage Scenarios

For each real-world scenario, identify the leakage mistake and the correct approach.

### Scenario 1: Hyperparameter Tuning on Test Set

A data scientist trains multiple models on training data but evaluates them on the same test set. They pick the model with the best test performance and report that as their final model's test accuracy.

1. **What is the leakage?**

   [Your explanation]

2. **Why is this a problem?**

   [Your reasoning]

3. **What should have been done?**

   [Correct approach]

---

### Scenario 2: Model Comparison Under Pressure

To decide between Model A and Model B for deployment, a team trains both on training data and evaluates both on test data multiple times, trying different random seeds. They report the best test score achieved.

1. **What is the leakage?**

   [Your explanation]

2. **Why does this advantage one model?**

   [Your reasoning]

3. **What is a fair comparison?**

   [Correct approach]

---

### Scenario 3: Target Leakage in Features

Someone builds a model to predict loan default using these features:
- Income
- Credit score
- Loan amount
- **Payment status in previous month** (0=on-time, 1=missed)

They split data randomly and get 95% accuracy. Great!

But then they realize: Payment status is known AFTER the outcome. If someone defaulted, their previous month payment status might reflect that.

1. **Why is this leakage?**

   [Your explanation]

2. **How did the split not catch this?**

   [Your reasoning]

3. **How should this feature be handled?**

   [Your solution]

---

## Part F: Cross-Validation vs Train-Test Split

### Task 1: Understanding the Difference

You have a dataset of 1000 examples.

**Scenario A: Train-Test Split**

```
Original Data (1000 examples)
│
├── Train (800)  ──→ Fit model  ──→ Single score
└── Test (200)   ──→ Evaluate  ──→ Report

Final accuracy reported: 85%
```

**Scenario B: 5-Fold Cross-Validation on Training Data**

```
Train Data (800)
├── Fold 1: Train on 640, Evaluate on 160  → Score: 84%
├── Fold 2: Train on 640, Evaluate on 160  → Score: 86%
├── Fold 3: Train on 640, Evaluate on 160  → Score: 87%
├── Fold 4: Train on 640, Evaluate on 160  → Score: 83%
└── Fold 5: Train on 640, Evaluate on 160  → Score: 85%

Mean CV score: 85%

Test Data (200)
└── Evaluate final model  → 82%
```

1. **What is cross-validation used for?**

   [Your explanation]

2. **Why don't we use cross-validation on the test set?**

   [Your reasoning]

3. **Which score do we report as our final model performance?**

   [Your answer]

4. **If CV score is 85% but test score is 60%, what does that indicate?**

   [Your interpretation]

---

## Part G: Verification Checklist

Create a verification script to check your project's split:

```python
import pandas as pd
from src.data_preprocessing import load_data, split_data

# Load and split
df = load_data('data/raw/delivery_data.csv')
X, y = df.drop('DeliveryDelayed', axis=1), df['DeliveryDelayed']
X_train, X_test, y_train, y_test = split_data(X, y)

# Verification checks
print("=" * 60)
print("TRAIN-TEST SPLIT VERIFICATION")
print("=" * 60)

# Check 1: Shapes
print("\n1. DATA SHAPES")
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")
print(f"   Total: {X_train.shape[0] + X_test.shape[0]} vs Original: {X.shape[0]}")
assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "Size mismatch!"
print("   ✓ Sizes correct")

# Check 2: No row overlap
print("\n2. OVERLAP CHECK")
try:
    overlap = len(set(X_train.index) & set(X_test.index))
    print(f"   Overlapping rows: {overlap}")
    assert overlap == 0, "Data leaked between sets!"
    print("   ✓ No overlap")
except:
    print("   ⚠ Could not check (may be using array indices)")

# Check 3: Class distribution
print("\n3. CLASS DISTRIBUTION")
if hasattr(y, 'value_counts'):
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    
    print("   Train:")
    for label, prop in train_dist.items():
        print(f"      Class {label}: {prop:.4f}")
    
    print("   Test:")
    for label, prop in test_dist.items():
        print(f"      Class {label}: {prop:.4f}")
    
    # Check difference
    max_diff = max(abs(train_dist[i] - test_dist[i]) for i in train_dist.index)
    print(f"   Max difference: {max_diff:.4f}")
    if max_diff < 0.05:
        print("   ✓ Stratification working (balanced distributions)")
    else:
        print("   ⚠ Distributions differ by more than 5%")

# Check 4: Data types
print("\n4. DATA TYPES")
print(f"   X_train types: {X_train.dtypes.unique()}")
print(f"   No NaN in X_train: {X_train.isnull().sum().sum() == 0}")
print(f"   No NaN in X_test: {X_test.isnull().sum().sum() == 0}")
print("   ✓ Data types valid")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
```

Run this script and capture the output:

**Your output:**

[Paste the output]

---

## Part H: Documentation

Write documentation for your project's data splitting strategy in a markdown file format:

```markdown
## Data Splitting Strategy

### Split Ratio
[Fill in your project's split]

### Method
[Stratified / Time-based / Random]

### Random State
[Your project's random state value]

### Class Distribution

| Set | Class 0 | Class 1 |
|-----|---------|---------|
| Train | [%] | [%] |
| Test | [%] | [%] |

### Preprocessing Order

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Leakage Prevention

Measures taken to prevent data leakage:
- [ ] Split before fitting any transformations
- [ ] Transformations fitted on training data only
- [ ] No feature selection on full dataset
- [ ] No hyperparameter tuning on test set
- [ ] Proper random state for reproducibility

### Verification Results

- Training set size: [N]
- Test set size: [N]
- Class balance preserved: Yes / No
- Documentation complete: Yes / No
```

---

## Part I: Reflection Questions

1. **Why is splitting before fitting transformations so critical?**

   [Your reflection]

2. **How would you explain to a colleague why evaluating on training data is misleading?**

   [Your explanation]

3. **If your model performs well on training data but poorly on test data, what might this indicate?**

   [Your interpretation]

4. **In your project, where are the boundaries between data that the model learns from vs data it's evaluated on?**

   [Your analysis]

5. **What would happen if you accidentally used test data during preprocessing?**

   [Your reasoning about consequences]

---

## Submission Checklist

- [ ] Part A: Verified your project's split strategy
- [ ] Part B: Identified leakage in incorrect approaches and wrote correct code
- [ ] Part C: Analyzed class balance and stratification in your project
- [ ] Part D: Decided on split strategy for 5 scenarios
- [ ] Part E: Analyzed real-world leakage scenarios
- [ ] Part F: Understood cross-validation vs train-test split
- [ ] Part G: Ran and documented verification script output
- [ ] Part H: Wrote documentation for your project's split strategy
- [ ] Part I: Answered reflection questions thoughtfully

---

## Learning Outcomes

After completing this assignment, you should understand:

✓ Why train-test splitting is essential for honest evaluation
✓ How to implement proper train-test splits with scikit-learn
✓ When to use stratified vs random vs time-based splitting
✓ The critical rule: split before fitting any transformations
✓ How to recognize and prevent data leakage
✓ How cross-validation differs from train-test evaluation
✓ How to verify splits are working correctly
✓ How to document splitting strategy for reproducibility

---

## Key Principle

**The train-test boundary is sacred.**

Once contaminated, evaluation becomes meaningless.

Split carefully. Validate the split. Protect the test set.

Honest evaluation is the foundation of trustworthy machine learning.
