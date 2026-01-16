"""
Multiple testing corrections.

Why this matters
----------------
In A/B testing, it is common to look at more than one metric.

Example:
- conversion rate
- revenue per visitor
- cancellation rate
- customer support contact rate

If you run many hypothesis tests at once and treat each p-value the same way
(for example, "p < 0.05 means significant"), then the chance of a false positive
increases as the number of tests grows.

This module provides standard corrections:

1) Bonferroni:
   Controls the probability of at least one false positive (family-wise error rate).
   Very simple, but can be overly conservative.

2) Holm–Bonferroni:
   Also controls family-wise error rate.
   Always at least as powerful as Bonferroni, while still being simple.

3) Benjamini–Hochberg (BH):
   Controls the expected proportion of false positives among the rejected tests
   (false discovery rate). Often used in metric dashboards.

All functions return adjusted p-values so you can compare them to your usual alpha.
"""

from __future__ import annotations

from typing import Iterable, List


def _to_pvalues(p_values: Iterable[float]) -> List[float]:
    p = [float(x) for x in p_values]
    if len(p) == 0:
        raise ValueError("p_values must contain at least 1 value.")
    for x in p:
        if x < 0.0 or x > 1.0:
            raise ValueError("All p-values must be between 0 and 1.")
    return p


def bonferroni(p_values: Iterable[float]) -> List[float]:
    """
    Bonferroni-adjust p-values.

    If m tests are performed, Bonferroni multiplies each p-value by m:

        p_adj = min(1, p * m)

    This controls the family-wise error rate.
    """
    p = _to_pvalues(p_values)
    m = len(p)
    return [min(1.0, x * m) for x in p]


def holm_bonferroni(p_values: Iterable[float]) -> List[float]:
    """
    Holm–Bonferroni adjusted p-values.

    Procedure (high level):
    1) Sort p-values from smallest to largest.
    2) Apply progressively less strict multipliers.
    3) Enforce monotonicity so adjusted p-values do not decrease as raw p-values increase.

    This controls the family-wise error rate and is always at least as powerful
    as the plain Bonferroni method.
    """
    p = _to_pvalues(p_values)
    m = len(p)

    indexed = list(enumerate(p))
    indexed.sort(key=lambda t: t[1])

    adjusted_sorted: List[float] = [0.0] * m
    for rank, (idx, value) in enumerate(indexed, start=1):
        multiplier = m - rank + 1
        adjusted_sorted[rank - 1] = min(1.0, value * multiplier)

    # Enforce monotonicity: adjusted p-values should be non-decreasing in sorted order.
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    # Map back to original order.
    adjusted = [0.0] * m
    for sorted_pos, (idx, _) in enumerate(indexed):
        adjusted[idx] = adjusted_sorted[sorted_pos]

    return adjusted


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    """
    Benjamini–Hochberg (BH) adjusted p-values.

    BH controls the false discovery rate (FDR).

    Procedure:
    1) Sort p-values ascending.
    2) Compute: p_adj(i) = p(i) * m / i  (where i is 1-based rank)
    3) Enforce monotonicity in reverse (so adjusted values do not decrease when moving
       from larger p-values to smaller p-values).
    4) Map back to original order.

    After adjustment, you can reject tests where p_adj <= alpha, where alpha is your FDR level.
    """
    p = _to_pvalues(p_values)
    m = len(p)

    indexed = list(enumerate(p))
    indexed.sort(key=lambda t: t[1])

    adjusted_sorted: List[float] = [0.0] * m
    for rank, (_, value) in enumerate(indexed, start=1):
        adjusted_sorted[rank - 1] = min(1.0, value * m / rank)

    # Enforce monotonicity from the end: q-values should not increase as rank decreases.
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted = [0.0] * m
    for sorted_pos, (idx, _) in enumerate(indexed):
        adjusted[idx] = adjusted_sorted[sorted_pos]

    return adjusted
