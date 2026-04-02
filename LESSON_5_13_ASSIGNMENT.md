# Lesson 5.13 Assignment: Understanding Supervised Learning Problem Types

## Overview

This assignment tests your understanding of problem type identification and the impact of choosing the right classification category, algorithm, and evaluation metrics. You will work through real-world scenarios and apply the concepts from the guide to actual data and business requirements.

---

## Part A: Problem Type Identification from Business Requirements

For each scenario, identify the problem type and justify your answer.

### Scenario 1: Bank Loan Default Prediction

**Business Need:** A bank wants to predict which loan applicants are likely to default (fail to repay) on their loans. This helps the bank decide whether to approve or deny a loan application.

**Questions to Answer:**

1. What is the target variable? What type of data is it?
2. What problem type is this? (Classification or Regression? Which subtype?)
3. What would be the positive class and negative class?
4. Why is accuracy a misleading metric for this problem?
5. Which evaluation metric should the bank prioritize and why?
6. Suggest three algorithms that would work well for this problem.

**Your Answer:**

[Write your answers here]

---

### Scenario 2: Real Estate Price Estimation

**Business Need:** A real estate company wants to estimate home prices for properties listed on their platform. This helps sellers set realistic asking prices and buyers understand if a price is competitive.

**Questions to Answer:**

1. What is the target variable? What type of data is it?
2. What problem type is this?
3. Why would it be wrong to convert prices to categories (Low, Medium, High)?
4. What evaluation metrics are appropriate for this problem?
5. What would success look like in business terms?
6. Suggest three algorithms that would work well for this problem.

**Your Answer:**

[Write your answers here]

---

### Scenario 3: Online Content Moderation

**Business Need:** A social media platform wants to automatically detect whether user-generated content violates community guidelines. Content can violate multiple guidelines (contains hate speech AND violence, or misinformation AND sexual content).

**Questions to Answer:**

1. What is the target variable(s)?
2. What problem type is this? Why?
3. Why is this fundamentally different from "categorize content as appropriate vs inappropriate"?
4. How would you structure the model(s) to handle this problem?
5. What metric would measure if predictions are getting the correct set of violations?
6. Why would Hamming loss be a meaningful metric here?

**Your Answer:**

[Write your answers here]

---

### Scenario 4: Medical Imaging Diagnosis

**Business Need:** A hospital wants to automatically detect tumors in medical scans. When a tumor is detected, it must be classified into one of four severity levels (None=0, Benign=1, Suspicious=2, Malignant=3).

**Questions to Answer:**

1. What is the target variable?
2. Is this binary classification, multi-class classification, or could it be regression?
3. What is special about this problem that makes it different from typical multi-class?
4. What is an important pitfall to avoid here?
5. Why might regression not be the right choice despite the ordinal nature?
6. What algorithm would be appropriate for this problem?

**Your Answer:**

[Write your answers here]

---

## Part B: Identifying Problem Types in Your Project

In the ML pipeline you built in Lesson 5.7, you trained a Random Forest model to predict delivery delays.

### Delivery Delay Prediction Analysis

1. **What is the target variable in your model?** Look at the data and describe it.

   ```python
   # Inspect the target variable in your dataset
   import pandas as pd
   df = pd.read_csv('data/raw/delivery_data.csv')
   print(df['DeliveryDelayed'].unique())
   print(df['DeliveryDelayed'].value_counts())
   ```

   **Your findings:**

   [Write what you found]

2. **Identify the problem type.** Is this classification or regression? Which subtype?

   **Your answer:**

   [Write your answer]

3. **Is the data balanced?** Calculate the percentage of each class.

   ```python
   # Calculate class distribution
   class_dist = df['DeliveryDelayed'].value_counts(normalize=True)
   print(class_dist)
   ```

   **Your findings:**

   [Write what you found]

4. **Which evaluation metric is most appropriate and why?** Consider whether accuracy is sufficient or if precision/recall/F1/ROC-AUC would be better.

   **Your reasoning:**

   [Write your reasoning]

5. **What would be the business impact of different types of errors?**

   - False Positive (predict late, actually on-time): Impact?
   - False Negative (predict on-time, actually late): Impact?

   **Your analysis:**

   [Write your analysis]

---

## Part C: Data Analysis and Metric Exploration

### Task: Analyze Your Project's Metrics in Context

In your project, you've already computed evaluation metrics (see `reports/metrics.json`). Now interpret them in the context of problem type.

1. **Load and display your current metrics:**

   ```python
   import json
   with open('reports/metrics.json', 'r') as f:
       metrics = json.load(f)
   
   for metric_name, metric_value in metrics.items():
       print(f"{metric_name}: {metric_value:.4f}")
   ```

   **Your metrics:**

   [Paste the output]

2. **Interpret each metric:**

   - Accuracy: What does this percentage mean in context of the business?
   - Precision: Of predictions that the model made as "Late," how many were actually late?
   - Recall: Of all the actually-late deliveries, what% did the model identify?
   - F1-Score: How does this capture the balance between precision and recall?

   **Your interpretation:**

   [Write your interpretation]

3. **If classes were severely imbalanced (95% on-time, 5% late), how would your interpretation change?**

   **Your answer:**

   [Write your answer]

---

## Part D: Pitfall Recognition

For each pitfall scenario below, identify:
1. What is the pitfall?
2. What would be the consequence?
3. How would you correct it?

### Pitfall 1: Converting Regression to Classification

Someone proposes converting house prices in your delivery dataset to categories: "Fast Delivery (<30 min)", "Medium Delivery (30-60 min)", "Slow Delivery (>60 min)". Then treating it as multi-class classification.

**Your analysis:**

[Write analysis of why this would be problematic if predicting continuous delivery time]

---

### Pitfall 2: Class Imbalance Ignored

In a fraud detection system, 99.5% of transactions are legitimate, 0.5% are fraudulent. A model that always predicts "legitimate" achieves 99.5% accuracy.

**Your analysis:**

1. Why is accuracy misleading here?
2. What metrics should be used instead?
3. What would be worst-case failure mode (what are we most concerned about)?

[Write your analysis]

---

### Pitfall 3: Multi-Label as Multi-Class

A movie recommendation system has genres: Action, Comedy, Drama, Horror, Romance, Thriller, Sci-Fi.

Someone builds a multi-class model that forces each movie into ONE genre.

**Your analysis:**

1. What information is lost?
2. How would predictions be wrong?
3. How should this problem be restructured?

[Write your analysis]

---

## Part E: Algorithm Selection by Problem Type

Match each problem type to appropriate algorithms. Some algorithms can apply to multiple types.

**Problem Types:**
- Binary Classification (Spam/Not Spam)
- Multi-Class Classification (Digit 0-9)
- Regression (House Price)
- Count Regression (Customer Purchases Next Month)

**Algorithms:**
- Logistic Regression
- Random Forest
- Linear Regression
- Naive Bayes
- Gradient Boosting
- Decision Trees
- Poisson Regression
- SVM

**Task:** Create a table showing which algorithms fit which problem types and explain why each is appropriate or inappropriate.

| Algorithm | Binary Classification | Multi-Class | Regression | Count Regression | Notes |
|-----------|----------------------|-------------|-----------|-----------------|-------|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| Linear Regression | | | | | |
| Naive Bayes | | | | | |
| Gradient Boosting | | | | | |
| Decision Trees | | | | | |
| Poisson Regression | | | | | |
| SVM | | | | | |

---

## Part F: Real-World Problem Type Scenarios

Work through these real-world scenarios and document your thinking.

### Scenario A: Predicting Student Test Performance

**Business Context:** A school wants to help struggling students by predicting their final exam score based on homework completion rate, attendance, prior test scores, and participation level.

**Your Analysis:**

1. Problem Type?
2. Target variable description?
3. Appropriate algorithms?
4. Key evaluation metric?
5. How would you handle if exam scores range from 0-100?

---

### Scenario B: Medical Diagnosis Assistant

**Business Context:** A hospital wants to help doctors by flagging patients who might have been exposed to infectious diseases. A patient might have exposure to COVID-19 AND influenza AND RSV simultaneously.

**Your Analysis:**

1. Problem Type?
2. Is this multi-class or multi-label? Why?
3. How many binary classifiers would you need?
4. What would Hamming loss measure?
5. How is this different from just predicting "Infectious" vs "Not Infectious"?

---

### Scenario C: Manufacturing Quality Control

**Business Context:** A factory counts defects in manufactured products. They want to predict how many defects will be found in tomorrow's production batch based on equipment settings, environmental conditions, and operator experience.

**Your Analysis:**

1. Why is this NOT standard regression?
2. Why is standard linear regression inappropriate?
3. What is special about count data?
4. What algorithm would be appropriate?
5. Why can't predictions be negative?

---

## Part G: Reflection Questions

Answer these reflection questions to solidify your understanding:

1. **Why does problem type matter before algorithm selection?**

   [Your reflection]

2. **How would you explain the difference between classification and regression to someone who has never studied ML?**

   [Your explanation]

3. **In your own words, why is class imbalance a pitfall in classification?**

   [Your explanation]

4. **Give an example from your own experience (or imagine one) where treating a regression problem as classification would be problematic.**

   [Your example]

5. **Why is it dangerous to use accuracy as the primary metric for class-imbalanced classification problems?**

   [Your explanation]

6. **How would you decide between precision and recall for a business problem you were assigned?**

   [Your decision process]

---

## Submission Checklist

- [ ] Part A: Problem type identification for all 4 scenarios completed with justification
- [ ] Part B: Analyzed delivery delay prediction in your project (balanced? appropriate metrics?)
- [ ] Part C: Loaded metrics from your project and interpreted them in context
- [ ] Part D: Identified pitfalls and explained consequences + corrections
- [ ] Part E: Algorithm/Problem Type matching table completed with notes
- [ ] Part F: Real-world scenarios (A, B, C) analyzed with full reasoning
- [ ] Part G: Reflection questions answered thoughtfully
- [ ] File saved and ready for review

---

## Learning Outcomes

After completing this assignment, you should be able to:

✓ Identify whether a supervised learning problem is classification or regression
✓ Determine classification subtype (binary, multi-class, multi-label)
✓ Recognize problem type from business requirements and raw data
✓ Select appropriate evaluation metrics based on problem type
✓ Identify common pitfalls in problem type analysis
✓ Choose algorithms appropriate for a given problem type
✓ Understand why problem type identification is foundational

---

## Next Steps

Once you understand problem types, you're ready to:
- Choose specific algorithms for your problem
- Train multiple models and compare them fairly
- Evaluate results with appropriate metrics
- Deploy models with realistic expectations about performance

Problem type is the foundation. Master this concept, and the rest follows naturally.
