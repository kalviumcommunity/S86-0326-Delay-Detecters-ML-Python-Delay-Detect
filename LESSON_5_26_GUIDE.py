"""
Lesson 5.26: Evaluating Classification Models Using Accuracy

Accuracy measures the proportion of correct predictions across all examples.
It is intuitive, fast to compute, and useful when classes are balanced and
error costs are symmetric.

=======================================================================
1) WHAT ACCURACY MEASURES
=======================================================================

Accuracy = (correct predictions) / (all predictions)

In binary confusion-matrix terms:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

It treats all correct predictions equally and all error types equally.

=======================================================================
2) WHEN ACCURACY WORKS
=======================================================================

Use accuracy as a primary metric when:
- class distribution is roughly balanced
- false positives and false negatives have similar cost
- overall correctness is the business objective

=======================================================================
3) WHEN ACCURACY MISLEADS
=======================================================================

On imbalanced data, a majority-class predictor can score very high accuracy
while failing to detect the minority class entirely.

Example:
- 95% class 0, 5% class 1
- always predict class 0 -> 95% accuracy, but 0% recall for class 1

Never trust accuracy alone in this setting.

=======================================================================
4) BALANCED ACCURACY
=======================================================================

Balanced accuracy averages recall across classes and is often a better default
for imbalanced classification.

Random binary classifier baseline for balanced accuracy is ~0.5.

=======================================================================
5) BASELINE COMPARISON
=======================================================================

Always compare model accuracy to majority-class baseline accuracy.

Questions to answer:
- Does model beat baseline?
- Is gain meaningful in practice?
- Is gain present on minority class, not only majority class?

=======================================================================
6) CONFUSION MATRIX CONNECTION
=======================================================================

Accuracy hides FP/FN trade-offs.
Always inspect confusion matrix and classification report to understand where
errors occur.

=======================================================================
7) CROSS-VALIDATION STABILITY
=======================================================================

Report mean +/- std for both:
- accuracy
- balanced accuracy

Use stratified folds so class ratios are preserved across folds.

=======================================================================
8) COMMON PITFALLS
=======================================================================

- reporting accuracy without baseline
- ignoring class imbalance
- skipping confusion matrix inspection
- treating tiny numeric gains as meaningful without business context
- evaluating only one split with no CV stability check

=======================================================================
KEY TAKEAWAY
=======================================================================

Accuracy is useful but not universal.
Use it with baseline comparison, balanced accuracy, confusion matrix insight,
and cross-validation to keep model evaluation honest.
"""


if __name__ == "__main__":
    print(__doc__)
