"""
Lesson 5.29: Creating and Interpreting a Confusion Matrix

The confusion matrix is the source table behind classification metrics.
It shows exactly where predictions are correct or incorrect.

=======================================================================
1) BINARY CONFUSION MATRIX LAYOUT
=======================================================================

For binary classification (scikit-learn ordering):

[[TN, FP],
 [FN, TP]]

Rows are actual labels, columns are predicted labels.

=======================================================================
2) CELL INTERPRETATION
=======================================================================

- TP: correctly detected positives
- TN: correctly rejected negatives
- FP: false alarms (predicted positive, actual negative)
- FN: missed positives (predicted negative, actual positive)

These four counts are sufficient to derive Accuracy, Precision, Recall,
F1, FPR, and FNR.

=======================================================================
3) METRICS FROM CELLS
=======================================================================

Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
FPR       = FP / (FP + TN)
FNR       = FN / (FN + TP)

Always verify denominators and handle zero-division safely.

=======================================================================
4) WHY NORMALIZATION MATTERS
=======================================================================

Raw counts can hide minority-class failure in imbalanced datasets.
Use row-normalization (normalize="true") so each row sums to 1 and reflects
class recall directly.

=======================================================================
5) THRESHOLD EFFECTS
=======================================================================

Lower threshold:
- TP tends to increase
- FP tends to increase
- FN tends to decrease

Higher threshold:
- TP tends to decrease
- FP tends to decrease
- FN tends to increase

Print confusion cells at multiple thresholds to make this trade-off explicit.

=======================================================================
6) MULTI-CLASS CONFUSION MATRICES
=======================================================================

For K classes, matrix shape is K x K.
- Diagonal: correct predictions by class
- Off-diagonal: class-pair confusions

Focus on the largest off-diagonal entries; they identify where feature
engineering or additional data may deliver the highest impact.

=======================================================================
7) COMMON MISTAKES
=======================================================================

- Misreading row/column meaning
- Reporting metrics without matrix counts
- Ignoring minority-class row
- Comparing raw matrices across different test-set sizes
- Tuning threshold on test data

=======================================================================
KEY TAKEAWAY
=======================================================================

Confusion matrices are not optional diagnostics. They are the primary evidence
for how a classifier behaves, where it fails, and which error type should be
optimized for your business objective.
"""


if __name__ == "__main__":
    print(__doc__)