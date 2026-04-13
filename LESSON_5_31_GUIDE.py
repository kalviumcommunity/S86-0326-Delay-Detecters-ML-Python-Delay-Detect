"""
Lesson 5.31: Understanding Bias and Variance Through Model Behavior

Bias and variance explain why models fail in predictable ways:
- high bias -> underfitting (both train and test perform poorly)
- high variance -> overfitting (train strong, test weak)

=======================================================================
1) CORE INTUITION
=======================================================================

Bias:
- error from overly restrictive assumptions
- model is too rigid to represent true patterns
- signature: low train performance and similarly low test performance

Variance:
- error from sensitivity to specific training samples
- model is too flexible and captures noise
- signature: strong train performance with weaker test performance

Good generalization requires balancing both.

=======================================================================
2) TRAIN VS TEST AS A DIAGNOSTIC
=======================================================================

Use train/test metrics as your first diagnostic:
- train low, test low, small gap -> high bias
- train high, test lower, large gap -> high variance
- train high, test high, small gap -> healthy fit

The train/test gap is the primary variance signal.

=======================================================================
3) COMPLEXITY MOVES THE BALANCE
=======================================================================

As complexity increases:
- bias tends to decrease
- variance tends to increase

As complexity decreases:
- bias tends to increase
- variance tends to decrease

This produces the classic U-shaped test error curve as model complexity
changes. There is usually an optimal middle region.

=======================================================================
4) LEARNING CURVES
=======================================================================

Learning curves plot train and validation scores versus training set size.

High bias pattern:
- train and validation converge quickly
- both plateau at modest performance
- adding data gives little gain

High variance pattern:
- train stays high
- validation is lower with a persistent gap
- adding data helps, but regularization/simplification may still be needed

=======================================================================
5) DATA SIZE EFFECT
=======================================================================

More data usually helps high variance more than high bias.
If model assumptions are wrong (bias problem), data alone cannot fix the
structural mismatch.

=======================================================================
6) PRACTICAL FIXES
=======================================================================

Reduce high bias:
- increase model flexibility
- engineer richer features
- reduce regularization strength
- lower K in KNN or allow deeper trees

Reduce high variance:
- regularize
- simplify model
- increase K in KNN or limit tree depth
- use feature selection
- add more training data when possible

=======================================================================
7) WHY THE TRADE-OFF IS UNAVOIDABLE
=======================================================================

Expected prediction error can be viewed as:
- bias^2 + variance + irreducible noise

Irreducible noise is a floor no model can remove. Tuning changes how
reducible error is split between bias and variance.

=======================================================================
KEY TAKEAWAY
=======================================================================

Do not optimize train score in isolation. Diagnose behavior using:
- train/test level
- train/test gap
- cross-validation mean and standard deviation
- learning curves

Then apply the fix that matches the diagnosed failure mode.
"""


if __name__ == "__main__":
    print(__doc__)
