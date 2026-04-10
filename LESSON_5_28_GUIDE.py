"""
Lesson 5.28: Evaluating Classification Models Using F1-Score

F1-score combines precision and recall into a single metric via harmonic mean.
It is especially useful when classes are imbalanced and both error types matter.

=======================================================================
1) DEFINITION
=======================================================================

F1 = 2 * (Precision * Recall) / (Precision + Recall)

Equivalent confusion-matrix form:
F1 = 2*TP / (2*TP + FP + FN)

True negatives are not part of F1.

=======================================================================
2) WHY HARMONIC MEAN
=======================================================================

The harmonic mean penalizes imbalance strongly.
A model cannot get high F1 by maximizing only precision or only recall.
Both must be reasonably high.

=======================================================================
3) WHEN F1 IS PREFERRED
=======================================================================

Use F1 as a primary metric when:
- classes are imbalanced
- positive-class quality and coverage both matter
- you need one summary metric for model selection

=======================================================================
4) WHEN F1 IS NOT ENOUGH
=======================================================================

F1 is not ideal when:
- one error type has much higher cost (use F-beta)
- true negatives matter strongly (consider balanced accuracy/MCC)
- threshold-independent ranking quality is required (consider PR-AUC/ROC-AUC)

=======================================================================
5) MULTI-CLASS AGGREGATION
=======================================================================

- micro F1: global aggregation, dominated by large classes
- macro F1: equal weight per class, best for minority visibility
- weighted F1: class-frequency weighted compromise

Choose averaging explicitly and justify it.

=======================================================================
6) THRESHOLD OPTIMIZATION
=======================================================================

Default threshold (0.5) is rarely optimal for F1.
Tune threshold on validation data, then evaluate once on test data.
Do not tune thresholds on test set.

=======================================================================
7) BASELINE CONTEXT
=======================================================================

Always compare model F1 to majority-class baseline F1.
Baseline minority-class F1 is often near zero.
Meaningful lift over this floor is required for deployment.

=======================================================================
8) STABILITY CHECK
=======================================================================

Use stratified cross-validation and report mean +/- std for F1.
Large variance indicates unstable subgroup behavior.

=======================================================================
KEY TAKEAWAY
=======================================================================

F1-score is often the most honest single-number metric for imbalanced
classification, but it should always be reported with precision, recall,
baseline comparison, threshold rationale, and CV stability.
"""


if __name__ == "__main__":
    print(__doc__)
