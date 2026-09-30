"""Borrowed base models and property functions from published repositories.

Nothing in this package is ours. Every module here adapts an *external*
checkpoint -- weights we did not train, on data we did not split -- to the
interfaces `guidance.py` and `sampling.py` already expect, so that the same
guidance arms run unchanged on somebody else's generator.

Why this package exists: our own guidance numbers are measured with our
generator, our guide `f_A` and our evaluator `f_B`. Even with the disjoint
`train_a` / `train_b` split, all three come from one pipeline, and a reviewer
is entitled to ask whether an arm's advantage is a property of the method or a
property of our stack. Re-running every arm on a base model and a property pair
that are both external answers that question, and it is the only way to answer
it without training another generator.

The adapters are deliberately thin. They do NOT reimplement the external
methods; they load the released weights, wrap them in our calling convention,
and leave the guidance code untouched.
"""
