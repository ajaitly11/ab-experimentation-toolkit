"""
Mean metric A/B testing (comparing averages)

This module helps answer questions like:

  "Did version B change the average value of a metric compared to version A?"

A "mean metric" is simply an average measured per user, such as:
- revenue per visitor (for each visitor, how much money they generated)
- time on page (for each visitor, how many seconds they stayed)
- items added to cart (for each visitor, how many items they added)

A realistic motivation: revenue per visitor
-------------------------------------------
Imagine an experiment on a booking website:

- Variant A (control): current booking flow
- Variant B (treatment): new booking flow

For each visitor, you record revenue in pounds.

Most visitors do not book at all, so their revenue is 0.
Some visitors book a hotel, so their revenue might be 80 or 150.
A few visitors book a more expensive trip, so their revenue might be 500 or more.

So the data is naturally noisy and uneven, which is normal in real product work.
Because of that noise, we should not judge success just by comparing two averages.
We need a method that accounts for variability and sample size.

What this module computes
-------------------------
The main function, `mean_diff(group_a, group_b)`, returns:

1) The average of group A and group B.
2) The effect:
       effect = mean(B) - mean(A)

   Example:
     mean(A) = 12.40
     mean(B) = 12.90
     effect  = 0.50

   This means B increased the average metric by about 0.50 units.

3) A 95% confidence interval for the effect.
   This gives a range of plausible values for the true effect, based on the sample.

   Example:
     effect = 0.50
     confidence interval = (0.10, 0.90)

   This can be read as:
     "Based on the data, the true lift is likely between 0.10 and 0.90."

   If the interval includes 0, then “no change” is still plausible.

4) A p-value for the hypothesis that the true effect is 0.
   A p-value answers:

     "If there were really no difference between A and B, how unusual would the
      observed difference be due to random chance?"

Why Welch's t-test
------------------
There are several ways to compare means. This toolkit uses Welch's t-test because it:
- does not assume both groups have the same variance (spread)
- works well even if group sizes differ a bit

Those are common conditions in real experiments.

Implementation note (kept simple)
---------------------------------
This module uses a normal-distribution approximation when converting the test statistic
into a p-value and confidence interval. For typical online experiments with hundreds
or thousands of observations per group, this behaves well and keeps the code lightweight.
"""

from __future__ import annotations
import math
from typing import Iterable, Sequence

from .types import MeanTestResult


def _to_floats(x: Iterable[float]) -> list[float]:
    """
    Convert input values into a list of floats.

    This makes the rest of the code simpler because:
    - we can compute length
    - we know the values are numeric
    - we can loop over them multiple times without exhausting an iterator

    If the input is empty, there is nothing to analyse.
    """
    xs = [float(v) for v in x]
    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")
    return xs


def _mean(xs: Sequence[float]) -> float:
    """
    Compute the average.

    Example:
      values = [10, 12, 9]
      mean   = (10 + 12 + 9) / 3 = 10.333...
    """
    return sum(xs) / len(xs)


def _sample_variance(xs: Sequence[float]) -> float:
    """
    Compute the unbiased sample variance.

    Variance measures how spread out the numbers are.

    If all values are identical, variance is 0.
    If values vary a lot, variance is larger.

    We use the common "unbiased" version that divides by (n - 1).

    If there is only one value, variance is not meaningful in the usual way.
    We return 0.0 so the rest of the calculations remain defined.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def _welch_degrees_of_freedom(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """
    Welch–Satterthwaite approximation for degrees of freedom.

    Degrees of freedom is a technical value used to map the test statistic to
    a probability. Welch's approach adjusts for unequal variances and sample sizes.

    In some degenerate cases (for example, both variances are zero), we return infinity,
    which effectively falls back to a normal distribution approximation.
    """
    a = var_a / n_a
    b = var_b / n_b
    numerator = (a + b) ** 2

    denominator = 0.0
    if n_a > 1 and var_a > 0:
        denominator += (a**2) / (n_a - 1)
    if n_b > 1 and var_b > 0:
        denominator += (b**2) / (n_b - 1)

    if denominator == 0.0:
        return float("inf")

    return numerator / denominator


def _normal_cumulative_distribution(z: float) -> float:
    """
    Cumulative distribution function for a standard normal distribution.

    Returns the probability that a value from Normal(0, 1) is <= z.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _approximate_t_cumulative_distribution(t_value: float, degrees_of_freedom: float) -> float:
    """
    Approximate cumulative distribution function for the t distribution.

    For large degrees of freedom, the t distribution is very close to the normal
    distribution, so we use the normal cumulative distribution function.

    In this module, this function is used with a non-negative t_value.
    """
    if degrees_of_freedom == float("inf") or degrees_of_freedom > 200:
        return _normal_cumulative_distribution(t_value)
    return _normal_cumulative_distribution(t_value)


def mean_diff(
    group_a: Iterable[float],
    group_b: Iterable[float],
    *,
    alpha: float = 0.05,
) -> MeanTestResult:
    """
    Compare the average of group B to the average of group A.

    Parameters
    ----------
    group_a, group_b:
        Metric values for each user in A and B.

        Example (time on page in seconds):
          group_a = [12, 35, 8, 20, 19, 5, 60]
          group_b = [15, 40, 9, 22, 25, 6, 70]

    alpha:
        Significance level for the confidence interval.
        alpha = 0.05 corresponds to a 95% confidence interval.

    Returns
    -------
    MeanTestResult
        Includes:
        - mean(A), mean(B)
        - effect = mean(B) - mean(A)
        - confidence interval for the effect
        - p-value for the hypothesis that the true effect is 0

    How the calculation works (step-by-step)
    ----------------------------------------
    1) Compute the two averages and their difference (the effect).
    2) Measure variability in each group using variance.
    3) Convert variability into a standard error for the difference in averages.
    4) Compute a test statistic: effect divided by standard error.
    5) Convert the test statistic into a two-sided p-value.
    6) Build a confidence interval using: effect ± critical value × standard error.
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

    # Standard error describes how much the estimated effect would typically fluctuate
    # due to random noise. Larger sample sizes reduce standard error.
    standard_error = math.sqrt(var_a / n_a + var_b / n_b)

    # Special case: if there is no variation in either group, the metric is constant.
    # Then the observed effect is deterministic.
    if standard_error == 0.0:
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

    test_statistic = effect / standard_error
    degrees_of_freedom = _welch_degrees_of_freedom(var_a, n_a, var_b, n_b)

    # Two-sided p-value: "extreme in either direction".
    cumulative_probability = _approximate_t_cumulative_distribution(
        abs(test_statistic), degrees_of_freedom
    )
    p_value = 2.0 * (1.0 - cumulative_probability)

    # 95% confidence interval uses a critical value close to 1.96 for the normal distribution.
    # This is the familiar "about 2 standard errors" rule of thumb.
    critical_value = 1.96
    ci_low = effect - critical_value * standard_error
    ci_high = effect + critical_value * standard_error

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
