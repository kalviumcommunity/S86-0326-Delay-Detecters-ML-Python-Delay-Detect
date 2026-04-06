"""
Lesson 5.16: Splitting Data into Training and Testing Sets

Before a model can be trusted, it must be evaluated on data it has never seen before.
This lesson explains why data splitting is essential, how to do it correctly,
and how to avoid common leakage mistakes that invalidate evaluation.

=======================================================================
1) WHY DATA SPLITTING IS NECESSARY
=======================================================================

Fundamental principle: If you train and evaluate on the same data, you measure
memorization, not learning.

Without proper splitting:
- Models memorize instead of generalizing
- Overfitting goes undetected
- Metrics appear artificially high
- Deployment performance collapses
- Evaluation becomes meaningless

The train-test split simulates the real-world scenario:
- Training data: Historical data the model learns from
- Testing data: Unseen examples it encounters during deployment

Test performance is a proxy for production performance.
If test performance is strong, you have evidence of generalization.
If it drops significantly vs training, you likely have overfitting.

CRITICAL INSIGHT:
    Train-test split is not optional. It is the only honest way to evaluate.

=======================================================================
2) TRAINING SET VS TESTING SET
=======================================================================

TRAINING SET
    Purpose:
    - Fit preprocessing transformations
    - Learn model parameters
    - Tune hyperparameters (via cross-validation)
    
    Key: Model is allowed to learn from this data

TESTING SET
    Purpose:
    - Evaluate final model performance
    - Compute unbiased metrics
    - Simulate unseen real-world inputs
    
    Key: Model must NEVER learn from this data
    
    The test set must remain untouched until final evaluation.
    Opening it early, using it for tuning, or peeking at results
    creates bias and invalidates performance estimates.

SEPARATION PRINCIPLE:
    What model sees during training → NEVER used for evaluation
    What model sees during testing → MUST be completely new

=======================================================================
3) STANDARD TRAIN-TEST SPLIT
=======================================================================

Most common split ratio:
    70-80% training
    20-30% testing

Why these ratios:
- Need enough training data to learn patterns
- Need enough test data to estimate generalization reliably
- 80-20 is standard and well-researched

Scikit-learn implementation:

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

IMPORTANT PARAMETERS:

test_size: Proportion of data used for testing
    - 0.2 means 20% testing, 80% training
    - Can also use absolute number: test_size=100 (100 examples in test)

random_state: Seed for reproducibility
    - Ensures same split every time code runs
    - CRITICAL for reproducible ML
    - Use fixed number like 42 (convention)

shuffle=True: (default)
    - Randomly shuffles before splitting
    - Essential for most datasets
    - Set to False only for specific cases (time-series)

REPRODUCIBILITY:
    Fixed random_state is non-negotiable.
    Different splits on different runs invalidate comparisons.
    
    # GOOD: Reproducible
    train_test_split(X, y, random_state=42)
    
    # BAD: Different split every run
    train_test_split(X, y)  # random_state defaults to None

=======================================================================
4) STRATIFIED SPLITTING FOR CLASSIFICATION
=======================================================================

Problem: In imbalanced classification, simple random split can distort
class distribution between train and test.

Example:
    Original data: 10% positive, 90% negative
    Random split might produce:
        Training: 12% positive, 88% negative
        Testing: 5% positive, 95% negative
    
    Different proportions = unstable evaluation metrics
    Misleading performance estimates

STRATIFIED SPLITTING:
    Ensures both sets maintain the same class proportions as original data.
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Preserves class balance
    )

Result:
    Original: 10% positive
    Training: 10% positive
    Testing: 10% positive
    
    Class distribution preserved across sets.

WHEN TO USE STRATIFICATION:

Classification: ALWAYS use stratified splitting
    - Binary classification: especially important if imbalanced
    - Multi-class classification: always use
    - Multi-label classification: can stratify on main labels

Regression: Stratification not applicable
    - Target is continuous, not discrete classes
    - Use standard train_test_split

VERIFICATION:

Compare class distributions:
    
    print("Original distribution:")
    print(y.value_counts(normalize=True))
    
    print("\nTrain distribution:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

All three should show same proportions.

=======================================================================
5) THE CRITICAL RULE: SPLIT BEFORE FITTING
=======================================================================

MOST IMPORTANT PRINCIPLE AT THE BOUNDARY OF TRAIN-TEST SPLIT:

Split the data BEFORE fitting any preprocessing transformations.

Why:
- Fitting transformations on full dataset leaks test data statistics
- Scaler parameters computed from test data influence training
- Models unknowingly learn test data patterns
- Evaluation becomes invalid

WRONG APPROACH (DATA LEAKAGE):

    # WRONG: Fitting before splitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit on ALL data (includes test)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    X_train_scaled = X_train  # Already scaled with test data in mind
    X_test_scaled = X_test

Problem: Scaler learned from test data statistics.
Result: Model performance appears better than it would in reality.

CORRECT APPROACH (NO LEAKAGE):

    # CORRECT: Splitting before fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
    X_test_scaled = scaler.transform(X_test)        # Transform test only

Result: Model never sees test data during fitting.
Evaluation is unbiased.

THE PATTERN:

    1. Split data first
    2. Fit transformers on TRAINING data
    3. Apply transformers to BOTH train and test
    4. Train model on transformed training data
    5. Evaluate on transformed test data

This pattern prevents leakage at every step.

COMMON TRANSFORMATION MISTAKES:

1) StandardScaler / MinMaxScaler
   Wrong: scaler.fit_transform(X), then split
   Right: split, then scaler.fit_transform(X_train), scaler.transform(X_test)

2) One-Hot Encoding
   Wrong: encode all categories in X, then split
   Right: split, then encode X_train, apply encoding to X_test
   
   Reason: If X_test has categories not seen in X_train,
   the encoder should handle that correctly.

3) Feature Selection
   Wrong: Select features from full dataset, then split
   Right: split, then select features from X_train
   
   Reason: Feature selection stats should be computed on train only.

4) Dimensionality Reduction (PCA, etc.)
   Wrong: Fit PCA on full X, then split
   Right: split, then fit PCA on X_train

=======================================================================
6) TIME-SERIES DATA: TEMPORAL SPLITTING
=======================================================================

For time-series or chronological data, DO NOT use random splitting.

Why:
- Random split violates temporal order
- Mixes future data into training
- Model uses future information to predict past
- Creates unrealistic, overly optimistic evaluation

Example: Predicting sales based on historical data
    
    Nov 2024 data: Use for training
    Dec 2024 data: Use for testing
    
    If randomly mixed:
        Training might contain Dec data
        Model learned from future → predicts past
        Evaluation misleading

CORRECT TIME-BASED SPLITTING:

    df = df.sort_values('Date')  # Ensure chronological order
    
    train_size = int(len(df) * 0.8)
    
    train_df = df.iloc[:train_size]      # Earlier 80%
    test_df = df.iloc[train_size:]        # Later 20%

Respects temporal order. Model trained on past, tested on future.

SCIKIT-LEARN ALTERNATIVE:

    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_index, test_index in tscv.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        # Train and evaluate

TimeSeriesSplit:
- Automatically respects order
- Useful for multiple folds
- Each test set is chronologically after all training

WHEN TO USE TIME-BASED SPLITTING:

Any time data has temporal order:
- Time-series data (stock prices, weather, etc.)
- Chronological data (logs, events, etc.)
- Sequences with order importance
- Any data where order matters

When NOT to use:
- Independent samples (images, medical records, transactions)
- Random order data (shuffled appropriately)

If unsure: Does the order matter for the problem?
    Yes → Use time-based splitting
    No → Use stratified random splitting

=======================================================================
7) DATA LEAKAGE BEYOND SPLITTING
=======================================================================

Train-test splitting prevents most leakage, but other mistakes remain:

LEAKAGE MISTAKE 1: Oversampling Before Splitting

Wrong:

    # WRONG: Oversample entire dataset first
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

Problem: SMOTE learned from all data (including test).
Result: Synthetic samples in training influenced by test set.

Correct:

    # CORRECT: Split first, oversample training only
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
    # X_test remains unchanged

Result: Training data balanced without leaking test information.

LEAKAGE MISTAKE 2: Feature Selection on Full Data

Wrong:

    # WRONG: Select features from all data
    important_features = select_k_best(X, y, k=10)
    X_selected = X[:, important_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

Problem: Feature selection learned from test data.

Correct:

    # CORRECT: Split first, select from training
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    important_features = select_k_best(X_train, y_train, k=10)
    X_train_selected = X_train[:, important_features]
    X_test_selected = X_test[:, important_features]

LEAKAGE MISTAKE 3: Hyperparameter Tuning on Test Set

Wrong:

    # WRONG: Tune hyperparameters by looking at test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    for param in param_range:
        model = Model(param=param)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)  # Tuning on test!
        best_param = param if score > best_score
    
    final_model = Model(param=best_param)
    final_model.fit(X_train, y_train)
    final_score = final_model.score(X_test, y_test)  # Biased if already tuned

Problem: Hyperparameters optimized for test set.
Result: Reported test performance is inflated.

Correct:

    # CORRECT: Hyperparameter tuning on training data via cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Tune on training data only
    grid_search = GridSearchCV(
        Model(),
        param_grid,
        cv=5  # Cross-validation on training data
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    final_score = best_model.score(X_test, y_test)  # Unbiased

Result: Hyperparameters optimized on training data, evaluated on test.

LEAKAGE MISTAKE 4: Multiple Comparisons / P-Hacking

Wrong:

    # WRONG: Try many models and report best test score
    for model_class in [LogisticRegression, SVM, RandomForest, GradientBoosting]:
        model = model_class()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_model = model
    
    report_best_model_performance()

Problem: Tested many models and reported best test result.
Result: Biased toward whichever model happened to score highest.

Correct:

    # CORRECT: Hold out separate validation set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    
    # Try models on validation set
    best_model = None
    best_val_score = 0
    for model_class in [LogisticRegression, SVM, RandomForest]:
        model = model_class()
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        if val_score > best_val_score:
            best_model = model
            best_val_score = val_score
    
    # Report test performance (which was never seen during selection)
    test_score = best_model.score(X_test, y_test)

Result: Fair comparison, unbiased test evaluation.

=======================================================================
8) VERIFYING THE SPLIT
=======================================================================

After splitting, verify:
1. Shapes are correct
2. No overlap between sets
3. Class distribution preserved (for classification)
4. No data corruption

VERIFICATION CODE:

    print("=== Data Shapes ===")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Total: {X_train.shape[0] + X_test.shape[0]} vs {X.shape[0]}")
    
    print("\n=== Class Distribution ===")
    print("Original:")
    print(y.value_counts(normalize=True))
    
    print("\nTrain:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest:")
    print(y_test.value_counts(normalize=True))
    
    print("\n=== Check for Overlap ===")
    # For regression: verify no overlap is harder
    # For classification: check class distributions differ between train/test only by random chance
    overlap_rate = len(set(X_train.index) & set(X_test.index)) / len(X_train)
    print(f"Overlap rate: {overlap_rate * 100:.2f}%")
    assert overlap_rate == 0, "Data leaked between train and test!"

TYPICAL EXPECTED RESULTS:

    Training set: (160, 8)
    Testing set: (40, 8)
    Total: 200 vs 200  # ✓ Correct

    Original:
    0    0.55
    1    0.45
    
    Train:
    0    0.5375
    1    0.4625
    
    Test:
    0    0.55
    1    0.45
    
    # ✓ Proportions preserved (within random variation)

=======================================================================
9) CROSS-VALIDATION VS TRAIN-TEST SPLIT
=======================================================================

These serve DIFFERENT purposes:

TRAIN-TEST SPLIT:
    - Single evaluation of final model
    - Holds out fixed test set
    - Simulates production deployment

CROSS-VALIDATION:
    - Multiple evaluations during development
    - Splits training data into k folds
    - Used for hyperparameter tuning and model selection
    - Reduces variance in performance estimate

Example:

    from sklearn.model_selection import cross_val_score
    
    # Cross-validation on training data ONLY
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5  # 5-fold cross-validation
    )
    print(f"CV scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f}")

Cross-validation process:
    1. Training data divided into 5 folds
    2. Train on 4 folds, evaluate on 1 fold
    3. Repeat 5 times (each fold as test once)
    4. Average the 5 scores

RELATIONSHIP:

    Train-test split (80-20):
    ├── Training data (80%)
    │   └── Used for cross-validation internally during model selection
    └── Test data (20%)
        └── Used only for final evaluation

DO NOT:
    - Use cross-validation on test set (defeats purpose)
    - Report cross-validation scores as final performance
    - Use test set for any intermediate evaluation

DO:
    - Use cross-validation on training data for hyperparameter tuning
    - Report test set performance as final model quality
    - Compare models using test set scores

=======================================================================
10) BEST PRACTICES: THE SPLITTING WORKFLOW
=======================================================================

CORRECT END-TO-END WORKFLOW:

    Step 1: Load full dataset
    X, y = load_data()
    
    Step 2: Split into train and test (FIRST)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # if classification
    )
    
    Step 3: Fit transformations on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    Step 4: Perform feature selection on training data only
    selector = SelectKBest(k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    Step 5: Hyperparameter tuning via cross-validation on training data
    grid_search = GridSearchCV(Model(), param_grid, cv=5)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    
    Step 6: Final evaluation on test data (NEVER BEFORE)
    test_score = best_model.score(X_test_selected, y_test)
    print(f"Final test performance: {test_score}")

CHECKLIST:

✓ Split data BEFORE any fitting
✓ Use stratified split for classification
✓ Use time-based split for temporal data
✓ Fixed random_state for reproducibility
✓ Fit transformations on training data only
✓ Apply transformations to both train and test
✓ Hyperparameter tune on training data (via CV)
✓ Report test performance as final result
✓ Never peek at test set until final evaluation
✓ Document split strategy in README

=======================================================================
11) DOCUMENTATION
=======================================================================

Example documentation in README or experiment log:

## Data Splitting Strategy

- **Dataset**: delivery_data.csv (1000 samples)
- **Split Ratio**: 80% training (800 samples), 20% testing (200 samples)
- **Stratification**: Yes, preserves class balance
- **Random State**: 42 (for reproducibility)

### Class Distribution

| Class | Original | Training | Testing |
|-------|----------|----------|---------|
| On-Time | 55% | 55.2% | 54.8% |
| Late | 45% | 44.8% | 45.2% |

### Preprocessing Order

1. Split data into train and test
2. Fit StandardScaler on training data
3. Apply scaler to both train and test
4. Train Random Forest on scaled training data
5. Evaluate on scaled test data

This transparency ensures anyone can reproduce the exact split.

=======================================================================
12) COMMON MISTAKES AND HOW TO AVOID THEM
=======================================================================

MISTAKE 1: Evaluating on training data
    Wrong: model.score(X_train, y_train) as final metric
    Why: Artificially inflated performance
    Fix: Always evaluate on held-out test set

MISTAKE 2: Scaling before splitting
    Wrong: StandardScaler().fit_transform(X) then split
    Why: Test data statistics affect training
    Fix: Split first, fit scaler on X_train

MISTAKE 3: Forgetting stratification
    Wrong: train_test_split(X, y) for classification
    Why: Class imbalance distorts evaluation
    Fix: train_test_split(X, y, stratify=y)

MISTAKE 4: Random split for time-series
    Wrong: train_test_split(time_series_data)
    Why: Violates temporal order, uses future to predict past
    Fix: Use time-based split: train_df = df[:n], test_df = df[n:]

MISTAKE 5: Using test set for hyperparameter tuning
    Wrong: GridSearchCV on test set
    Why: Overfits hyperparameters to test data
    Fix: Use cross-validation on training data

MISTAKE 6: Oversampling before splitting
    Wrong: SMOTE(X, y).fit_resample(), then split
    Why: Test data influences synthetic training samples
    Fix: Split first, oversample training data only

MISTAKE 7: Multiple comparisons
    Wrong: Try 50 models, report best test score
    Why: Biased toward lucky models
    Fix: Use held-out validation set for selection, test set for reporting

=======================================================================
13) MENTAL MODEL
=======================================================================

Remember:

TRAIN-TEST SPLIT IS A BARRIER.

    ┌─────────────────────────────────────────────┐
    │ Original Data                               │
    ├──────────────────────┬──────────────────────┤
    │ Training Set (80%)   │ Test Set (20%)       │
    │                      │ [SEALED UNTIL END]   │
    │ Fit transforms here  │                      │
    │ Train model here     │                      │
    │ Tune parameters here │                      │
    │ (via cross-val)      │ Evaluate here        │
    │                      │ Report result        │
    └──────────────────────┴──────────────────────┘

Nothing crosses that barrier until final evaluation.

GOLDEN RULE:

    If data is used to make any decision
    (fitting, tuning, selecting),
    it cannot be used to evaluate.

Split carefully. Protect the test set. Trust the evaluation.

=======================================================================
"""

if __name__ == "__main__":
    print(__doc__)
