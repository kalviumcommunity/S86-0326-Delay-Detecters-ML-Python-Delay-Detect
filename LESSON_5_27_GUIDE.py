"""
Lesson 5.27: Evaluating Classification Models Using Precision and Recall

Precision and Recall evaluate different aspects of positive-class performance:
- Precision: quality of positive predictions
- Recall: completeness of positive detection

=======================================================================
1) CORE DEFINITIONS
=======================================================================

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)

Precision asks: "When we predict positive, how often are we right?"
Recall asks: "Of actual positives, how many did we catch?"

=======================================================================
2) WHY ACCURACY IS INSUFFICIENT
=======================================================================

Accuracy can be high on imbalanced data even when minority-class detection is
poor. Precision/Recall expose this failure directly.

=======================================================================
3) TRADE-OFF MECHANISM
=======================================================================

Changing classification threshold shifts trade-off:
- higher threshold -> usually higher precision, lower recall
- lower threshold -> usually higher recall, lower precision

No single threshold is universally correct; choose based on cost asymmetry.

=======================================================================
4) WHEN TO PRIORITIZE EACH
=======================================================================

Prioritize Precision when false positives are costly.
Prioritize Recall when false negatives are costly.

Examples:
- Spam filtering: precision-centric
- Disease/fraud screening: recall-centric

=======================================================================
5) BASELINE COMPARISON
=======================================================================

Always compare to majority-class baseline.
For minority class, baseline recall is often zero.
A useful model must materially exceed this floor.

=======================================================================
6) F1 AND F-BETA
=======================================================================

F1 balances precision and recall via harmonic mean.
Use F-beta when one metric should dominate:
- F2 emphasizes recall
- F0.5 emphasizes precision

=======================================================================
7) STABILITY WITH CROSS-VALIDATION
=======================================================================

Report mean +/- std for precision, recall, and F1 over stratified folds.
High variance indicates unstable subgroup performance risk.

=======================================================================
8) COMMON PITFALLS
=======================================================================

- optimizing only precision or only recall in isolation
- leaving threshold at 0.5 without business justification
- reporting metrics without confusion matrix and baseline context
- ignoring minority-class rows in classification report

=======================================================================
KEY TAKEAWAY
=======================================================================

Precision and Recall are the core metrics for high-stakes and imbalanced
classification tasks. Use both, tune threshold intentionally, and validate
stability before deployment.
"""


if __name__ == "__main__":
    print(__doc__)
