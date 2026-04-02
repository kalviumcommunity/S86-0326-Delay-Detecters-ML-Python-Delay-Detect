"""
Lesson 5.13: Understanding Supervised Learning Problem Types

Before training any model, you must understand what kind of problem you are solving.
This lesson builds the conceptual framework for identifying and working with different
supervised learning problem types systematically.

=======================================================================
1) WHAT IS SUPERVISED LEARNING?
=======================================================================

Supervised learning: Training on labeled data where each example has both
input features (X) and correct output (y).

Formula:
    Input Features (X) + Correct Answers (y) → Training → Model → Predictions

Key distinction:
- Supervised: You have labeled data (correct answers known)
- Unsupervised: Only input data, no labels (clustering, dimensionality reduction)
- Reinforcement: Agent learns by reward/penalty

In this sprint: Supervised learning only.
You have datasets where target variable is known for historical examples.
Your job: Build model to predict it for future examples.

=======================================================================
2) THE TWO CORE PROBLEM TYPES
=======================================================================

All supervised learning divides into two categories based on target type:

CLASSIFICATION: Predicting Categories
    Output is discrete — a label from a fixed set
    
REGRESSION: Predicting Continuous Values
    Output is a number on a continuous scale

The distinction determines:
- Which algorithms are appropriate
- Which metrics are meaningful
- How success is defined

=======================================================================
3) CLASSIFICATION: PREDICTING CATEGORIES
=======================================================================

Output: A label from a fixed set of possibilities. No notion of "how much."
An email is spam OR not spam. A flower is setosa, versicolor, OR virginica.

Examples:
- Email: Spam vs Not Spam (Binary)
- Flower species: Setosa, Versicolor, or Virginica (Multi-class)
- Patient diagnosis: Disease present or absent (Binary)
- Movie genres: Action, Comedy, Drama (Multi-label)

Model outputs:
- Predicted class label
- Probability distribution over classes

Binary classification example:
    Prediction: Spam
    Probabilities: {Not Spam: 0.15, Spam: 0.85}

This tells you not just the prediction, but confidence level.
An 85% confidence is different from 51%, even though both predict Spam.

BINARY CLASSIFICATION
- Exactly two mutually exclusive classes
- Positive class (1): Thing you're looking for (spam, fraud, disease)
- Negative class (0): Absence of that thing (not spam, legitimate, healthy)
- Model outputs single probability: probability of positive class
- Threshold (typically 0.5) decides classification

Common challenge: Class imbalance
    If 99% examples are negative, model that always predicts negative
    achieves 99% accuracy while being useless. Accuracy is misleading.
    Solution: Use precision, recall, F1, ROC-AUC instead.

MULTI-CLASS CLASSIFICATION
- Three or more mutually exclusive classes
- Each example belongs to exactly one class
- Examples:
    - Iris: Setosa, Versicolor, Virginica
    - Digit recognition: 0, 1, 2, ..., 9
    - Document categories: Sports, Politics, Tech, Entertainment

Model output: Probability distribution over all classes summing to 1.0

Example:
    Predicted probabilities: {Setosa: 0.05, Versicolor: 0.80, Virginica: 0.15}
    Predicted class: Versicolor (highest probability)

Class imbalance common here too:
    If 80% examples are one class, models learn to predict majority class.
    Solution: Use macro/micro/weighted averaging for metrics.

MULTI-LABEL CLASSIFICATION
- Multiple labels can apply simultaneously
- Not mutually exclusive like multi-class
- Examples:
    - Movie genres: A movie can be {Action, Comedy}
    - Patient diagnosis: Patient can have {Diabetes, Hypertension, Asthma}
    - Document tags: Article can be {ML, Python, Tutorial}

Model output: Separate binary probability for each label (independent)

Example:
    Person: 0.95 (Yes)
    Dog: 0.88 (Yes)
    Beach: 0.12 (No)
    Sunset: 0.67 (Yes)
    Predicted labels: {Person, Dog, Sunset}

Often decomposed into multiple binary problems (one per label).
Evaluation complexity: How well each label AND overall combinations predicted.

=======================================================================
4) REGRESSION: PREDICTING CONTINUOUS VALUES
=======================================================================

Output: A number on continuous scale. Magnitude and distance matter.
Price $310,000 is closer to $285,000 than to $500,000.

Examples:
- House price prediction: $285,000
- Stock price: $125.75
- Temperature: 23.5°C
- Delivery time: 37 minutes
- Salary: $85,000

Model outputs:
- Single numerical prediction
- Optionally: confidence interval or prediction interval

Example:
    Prediction: $285,000
    95% Confidence Interval: [$270,000 - $300,000]

SIMPLE LINEAR REGRESSION
    Model: y = mx + b
    Predicts output as linear function of one input feature
    
    Example: House price based only on square footage
    Real-world: rarely this simple, mostly for teaching

MULTIPLE LINEAR REGRESSION
    Model: y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
    Predicts output as linear combination of multiple inputs
    
    Example: House price based on sqft, bedrooms, location, age
    
    Assumes linear relationships (often violated in real data)
    More complex models often perform better

NON-LINEAR REGRESSION
    Models complex, non-linear relationships
    Approaches:
    - Polynomial regression (transform features: x, x², x³)
    - Tree-based (decision trees, random forests, gradient boosting)
    - Neural networks (arbitrary non-linear functions)
    
    More flexible but prone to overfitting

COUNT REGRESSION
    Special case: Predict non-negative integers (counts)
    
    Examples:
    - Customer purchases next month
    - Website visits per day
    - Insurance claims
    - Manufacturing defects
    
    Properties:
    - Cannot be negative (0, 1, 2, 3, ...)
    - Many zeros common
    - Variance often increases with mean
    
    Specialized models:
    - Poisson regression
    - Negative binomial regression

=======================================================================
5) HOW TO IDENTIFY PROBLEM TYPE
=======================================================================

When given a business task, translate from business language to ML language.

QUESTION 1: What am I predicting?

Output is a category from fixed set?
    → Classification

Output is a number on continuous scale?
    → Regression

Examples:
    "Predict which customers will churn" → Yes/No → Binary Classification
    "Predict how much revenue each customer generates" → Dollar amount → Regression
    "Predict product category user will purchase" → Category → Multi-class
    "Predict diabetes probability" → If binary Yes/No → Classification

QUESTION 2: How many possible outcomes?

For classification:
    Two classes → Binary classification
    Three+ mutually exclusive → Multi-class
    Multiple labels apply simultaneously → Multi-label

For regression:
    Unbounded continuous → Standard regression
    Non-negative integers → Count regression
    Bounded range (0-100 percentage) → Regression + transformation

QUESTION 3: What does target variable look like?

Look at actual data:

    # Classification target
    df['Churn'].unique()
    → array(['Yes', 'No'])

    # Regression target
    df['Price'].head()
    → [180000, 235000, 195000, 275000, 189500]

QUESTION 4: What does success look like?

Business success criterion determines technical metric:

    "Catch as many fraud cases as possible" → Classification, optimize recall
    "Predictions within $10K of actual price" → Regression, optimize MAE
    "Rank customers by churn likelihood" → Classification, optimize ROC-AUC

=======================================================================
6) CHOOSING ALGORITHMS BY PROBLEM TYPE
=======================================================================

Problem type constrains algorithm choices.

BINARY CLASSIFICATION:
- Logistic Regression (simple, interpretable, fast)
- Decision Trees (non-linear, interpretable, prone to overfit)
- Random Forest (ensemble, reduces overfitting)
- Gradient Boosting (powerful, handles complex patterns)
- SVM (works well in high dimensions)
- Naive Bayes (fast, good for text)

MULTI-CLASS CLASSIFICATION:
- Most binary classifiers extend to multi-class
- Logistic Regression (Softmax)
- Random Forest
- Gradient Boosting
- Neural networks with softmax

REGRESSION:
- Linear Regression (simple, assumes linearity)
- Ridge / Lasso Regression (linear + regularization)
- Decision Tree Regressor (non-linear, interpretable)
- Random Forest Regressor (ensemble)
- Gradient Boosting Regressor (powerful)
- Neural Networks (flexible, learns arbitrary functions)

=======================================================================
7) EVALUATION METRICS BY PROBLEM TYPE
=======================================================================

Using wrong metric can lead optimization for wrong thing.

CLASSIFICATION METRICS:

Accuracy: Fraction of correct predictions
    Misleading when imbalanced (99% negative, 1% positive)

Precision: Of predicted positives, what fraction actually positive?
    Low false positives. "How many spam predictions are correct?"

Recall (Sensitivity): Of actual positives, what fraction did we catch?
    Low false negatives. "How many spam emails did we correctly identify?"

F1 Score: Harmonic mean of precision and recall
    Balances both. Good for imbalanced data.

ROC-AUC: Area under receiver operating characteristic curve
    Threshold-independent. Good for ranking quality.

Confusion Matrix: Visualization of TP, FP, TN, FN

REGRESSION METRICS:

MAE (Mean Absolute Error): Average absolute difference
    Interpretable, robust to outliers
    "Predictions off by average $X"

MSE (Mean Squared Error): Average squared difference
    Penalizes large errors more heavily

RMSE (Root Mean Squared Error): Square root of MSE
    Same units as target variable

R² (Coefficient of Determination): Variance explained
    0 = no better than mean, 1 = perfect

MAPE (Mean Absolute Percentage Error): Average percentage error
    Interpretable but sensitive to small true values

=======================================================================
8) COMMON PITFALLS
=======================================================================

PITFALL 1: Treating regression as classification unnecessarily
    Mistake: Bin continuous prices into {Low, Medium, High}, use classification
    Problem: Lose information. $285K and $310K both "High," but close together.
    When appropriate: Business only cares about broad categories, not precision.

PITFALL 2: Treating classification as regression
    Mistake: Encode categories as numbers (Setosa=1, Versicolor=2) and regress
    Problem: Numbers imply ordering/magnitude that don't exist.
    Exception: Ordinal classification (ratings with natural order)

PITFALL 3: Ignoring class imbalance
    Mistake: 99% negative class, model always predicts negative, gets 99% accuracy
    Problem: Accuracy misleading. Model learned nothing.
    Solution: Use precision, recall, F1 score, ROC-AUC. Check confusion matrix.

PITFALL 4: Confusing multi-class and multi-label
    Mistake: Movie can be Action AND Comedy, force model to pick one
    Problem: Lose information. Predictions incomplete.
    Solution: Recognize non-mutually exclusive labels. Use multi-label techniques.

PITFALL 5: Wrong evaluation metric
    Mistake: Optimize accuracy on imbalanced binary classification
    Problem: Model learns to predict majority class and be "accurate" without learning
    Solution: Use metric aligned with business goal (precision, recall, F1, ROC-AUC)

=======================================================================
9) REAL-WORLD EXAMPLES
=======================================================================

EXAMPLE 1: EMAIL SPAM DETECTION
    Target: Is this email spam? (Yes/No)
    Problem type: Binary classification
    Algorithms: Logistic Regression, Naive Bayes, Random Forest, SVM
    Metrics: Precision high (avoid false positives), Recall high (catch spam)
    Imbalance: Usually yes, most emails not spam

EXAMPLE 2: HOUSE PRICE PREDICTION
    Target: Sale price ($285,000)
    Problem type: Regression
    Algorithms: Linear Regression, Ridge/Lasso, Random Forest, Gradient Boosting
    Metrics: RMSE, MAE, R²
    Success: Predictions within certain dollar amount or percentage

EXAMPLE 3: CUSTOMER CHURN PREDICTION
    Target: Will customer churn in next 30 days? (Yes/No)
    Problem type: Binary classification
    Algorithms: Logistic Regression, Random Forest, Gradient Boosting
    Metrics: High recall (catch at-risk customers), acceptable precision
    Imbalance: Usually yes, most customers don't churn

EXAMPLE 4: HANDWRITTEN DIGIT RECOGNITION
    Target: Which digit? (0-9)
    Problem type: Multi-class classification (10 classes)
    Algorithms: Neural Networks, Random Forest, SVM
    Metrics: Accuracy (if balanced), Confusion matrix, Macro F1

EXAMPLE 5: MOVIE GENRE TAGGING
    Target: Which genres apply? (Action, Comedy, Drama, etc.)
    Problem type: Multi-label classification
    Algorithms: Binary classifier per genre, Neural networks
    Metrics: Hamming loss, Subset accuracy, Micro/Macro F1

=======================================================================
10) MENTAL MODEL
=======================================================================

Remember:

Classification: Output is a category from a fixed set
    → Use classification algorithms
    → Evaluate with accuracy, precision, recall, F1, ROC-AUC
    → Watch out for class imbalance

Regression: Output is a continuous number
    → Use regression algorithms
    → Evaluate with MAE, RMSE, R²
    → Predictions are interpretable quantities (dollars, degrees, etc.)

Choose problem type FIRST.
Then choose algorithm.
Then choose evaluation metric.
Then train and evaluate.

Get problem type right, and everything else follows.
Get it wrong, and all subsequent work is misguided.

=======================================================================
KEY TAKEAWAY
=======================================================================

Before training any model:

Ask: What am I predicting?
    - Category or continuous value?
    - How many outcomes?
    - Mutually exclusive or multiple?

Look at your data:
    - What type is the target variable?
    - What are the values?
    - Are classes balanced?

Define success:
    - What does "good" look like for this business problem?
    - Which metric aligns with that definition?

THEN train the model.

Problem type is the foundation.
Everything builds from there.
"""

if __name__ == "__main__":
    print(__doc__)
