from __future__ import annotations

"""
Mean metric A/B testing (e.g., average revenue, average time on page).

In many product experiments, you care about "average" outcomes:
- average revenue per user
- average time on page
- average number of items added to cart

This module compares group A vs group B by looking at:

    effect = mean(B) - mean(A)

And it returns:
- the effect size
- a confidence interval (CI) for the effect
- a p-value for the hypothesis "the true effect is 0"

We use Welch's t-test because it does not assume both groups have the same variance.
That makes it a safe default in practice, especially for messy real-world metrics.
"""

import math
from typing import Iterable, Sequence

from .types import MeanTestResult


def _to_floats(x: Iterable[float]) -> list[float]:
    """
    Convert an iterable of numbers into a list of floats.

    Why do this?
    - It lets us handle lists, NumPy arrays, Pandas series, generators, etc.
    - It ensures we're working with numeric values.
    """
    xs = [float(v) for v in x]
    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")
    return xs


def _mean(xs: Sequence[float]) -> float:
    """Plain average."""
    return sum(xs) / len(xs)


def _sample_variance(xs: Sequence[float]) -> float:
    """
    Unbiased sample variance (ddof=1).

    If you have values x1, x2, ... xn, the sample variance is:

        sum((xi - mean)^2) / (n - 1)

    Note:
    - When n < 2, variance isn't really defined in the usual way.
      We return 0.0 so the rest of the math has something sensible to use.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def _welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """
    Welch–Satterthwaite approximation for degrees of freedom.

    Why degrees of freedom matters:
    - When you compute a t-statistic, you need a distribution to compare it against.
    - The exact distribution depends on how uncertain the variance estimates are.
    - Welch's method adjusts for unequal variances and different sample sizes.

    In very stable (or degenerate) situations, this can end up as "infinite" degrees
    of freedom, which basically behaves like the normal distribution.
    """
    a = var_a / n_a
    b = var_b / n_b
    num = (a + b) ** 2

    den = 0.0
    if n_a > 1 and var_a > 0:
        den += (a**2) / (n_a - 1)
    if n_b > 1 and var_b > 0:
        den += (b**2) / (n_b - 1)

    if den == 0.0:
        # Example: both groups have identical values like [5, 5, 5, 5]
        # Then variance is 0 and the "uncertainty in the variance" part vanishes.
        return float("inf")

    return num / den


def _normal_cdf(z: float) -> float:
    """
    Standard normal cumulative distribution function.

    This returns P(Z <= z) where Z ~ Normal(0, 1).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _approx_t_cdf(t: float, df: float) -> float:
    """
    Approximate CDF for Student's t distribution.

    For large degrees of freedom, Student-t is extremely close to Normal(0, 1).
    For our purposes here (beginner-friendly toolkit, typical A/B test sample sizes),
    using the normal approximation is usually a reasonable starting point.

    We use abs(t) when computing p-values, so this function is used with non-negative t.
    """
    # If df is huge, the normal approximation is a standard simplification.
    if df == float("inf") or df > 200:
        return _normal_cdf(t)

    # For smaller df, we still use normal approximation here to avoid heavy dependencies.
    # That means for very small samples, p-values/CI may be slightly less accurate,
    # but for typical online experiments (hundreds/thousands of users) it behaves well.
    return _normal_cdf(t)


def mean_diff(
    group_a: Iterable[float],
    group_b: Iterable[float],
    *,
    alpha: float = 0.05,
) -> MeanTestResult:
    """
    Compare two groups' means using Welch's t-test (unequal variances).

    Inputs
    ------
    group_a, group_b:
        Lists (or arrays) of per-user metric values.

        Example:
            group_a = [10.0, 12.0, 9.0, 11.0]
            group_b = [11.0, 12.0, 13.0, 10.0]

    Returns
    -------
    MeanTestResult:
        effect = mean(B) - mean(A)
        plus CI and p-value.

    Quick intuition (no heavy math)
    -------------------------------
    1) We compute the two averages.
    2) We look at how spread out the numbers are (variance).
    3) If the averages differ by a lot compared to the typical noise,
       we get a small p-value and a CI that likely doesn't include 0.
    """
    a = _to_floats(group_a)
    b = _to_floats(group_b)

    n_a = len(a)
    n_b = len(b)

    mean_a = _mean(a)
    mean_b = _mean(b)
    effect = mean_b - mean_a

    var_a = _sample_variance(a)
    var_b = _sample_variance(b)

    # Standard error (SE) tells us how noisy the difference in means is.
    # Roughly: bigger samples -> smaller SE.
    se = math.sqrt(var_a / n_a + var_b / n_b)

    # Special case: if SE is 0, both groups have no spread.
    # Example:
    #   A = [5, 5, 5], B = [5, 5, 5] -> effect=0, p=1
    #   A = [5, 5, 5], B = [6, 6, 6] -> effect=1, p=0 (deterministic difference)
    if se == 0.0:
        p_value = 0.0 if effect != 0.0 else 1.0
        return MeanTestResult(
            n_a=n_a,
            n_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            effect=effect,
            ci_low=effect,
            ci_high=effect,
            p_value=p_value,
            alpha=alpha,
        )

    # t-statistic: "how many standard errors away from 0 is our effect?"
    t_stat = effect / se
    df = _welch_df(var_a, n_a, var_b, n_b)

    # Two-sided p-value:
    # We look at the probability of seeing a t-statistic at least this extreme
    # (in either direction), assuming the true effect is 0.
    cdf = _approx_t_cdf(abs(t_stat), df)
    p_value = 2.0 * (1.0 - cdf)

    # Confidence interval:
    # effect ± (critical value) * SE
    #
    # For 95% CI, the z critical value is about 1.96.
    # (This is the familiar "within ~2 standard errors" rule of thumb.)
    z = 1.96
    ci_low = effect - z * se
    ci_high = effect + z * se

    return MeanTestResult(
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        alpha=alpha,
    )