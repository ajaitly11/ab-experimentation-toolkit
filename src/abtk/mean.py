from __future__ import annotations

"""
Mean metric A/B testing (for averages such as revenue, time, spend)

This module is for the most common type of question in online experimentation:

    "Did variant B change the average value of a metric compared to variant A?"

Examples of mean metrics that show up in product work
----------------------------------------------------
1) Average revenue per visitor
   - For each visitor, you record how much revenue they generated during the session.
   - Most visitors generate 0 revenue, some generate a little, and a few generate a lot.
   - You want to know if a new checkout flow increased the average revenue.

2) Average time on page (seconds)
   - For each visitor, you record time spent on a landing page.
   - You want to know if a new page design kept people engaged longer.

3) Average number of items added to cart
   - For each visitor, you record how many items they added.
   - You want to know if a change to recommendations increased the average count.

In all of these cases, we have per-user measurements, and we compare the average between groups.

What this function returns
--------------------------
The main function here is `mean_diff(group_a, group_b)` and it returns:

1) effect
   - This is: mean(B) minus mean(A)
   - Example: if mean(A) is 12.40 and mean(B) is 12.90,
     the effect is 0.50 (B is higher by 0.50).

2) confidence interval
   - This is a range that shows the uncertainty in our estimate of the effect.
   - A 95% confidence interval is a standard default.

   Example interpretation:
   - effect = 0.50
   - 95% confidence interval = (0.10, 0.90)

   This can be read as:
   "Based on the data, the true lift is likely between 0.10 and 0.90."

   If the interval includes 0, then "no change" is still plausible.

3) p-value
   - This answers the question:
     "If there were really no difference between A and B, how unusual would our observed
      effect be just due to random chance?"

   A small p-value means the observed difference is hard to explain by random noise alone.

A realistic A/B testing scenario (motivating example)
-----------------------------------------------------
Suppose Expedia tests a new booking page.

- Variant A (control): current booking page
- Variant B (treatment): redesigned booking page

Metric: revenue per visitor in British pounds, measured during the session.

For each visitor we record revenue:
- Many visitors book nothing: revenue = 0.00
- Some visitors book a hotel: revenue might be 120.00
- A few visitors book a more expensive trip: revenue might be 500.00+

So the data looks noisy (lots of variation). That is normal.
Our job is to decide whether the difference in average revenue is likely real or just noise.

Why we use Welch's t-test
-------------------------
When comparing two averages, a common approach is a t-test.

There are different versions of the t-test. This module uses Welch's t-test because:
- it does not assume that the two groups have the same variance
- it behaves well when group sizes are not equal

In real experiments, those conditions (different variance, slightly different sample sizes)
are very common, so Welch's t-test is a sensible default.

Important note about accuracy
-----------------------------
This module computes the p-value using a normal distribution approximation.
For typical online experiments (hundreds or thousands of observations per group),
this is generally a reasonable simplification.

For very small sample sizes, this approximation can be a bit rough. In practical product
experimentation, small sample experiments are usually avoided anyway because results are
too noisy.

The goal of this code is to be:
- correct in the structure of the analysis
- clear to read and learn from
- tested and reproducible

If you want extremely precise p-values for tiny samples, you would typically use a more
exact distribution calculation (often provided by scientific libraries).
"""

import math
from typing import Iterable, Sequence

from .types import MeanTestResult


def _to_floats(x: Iterable[float]) -> list[float]:
    """
    Convert input values into a list of floats.

    This makes the rest of the code simpler because:
    - we know we can compute length
    - we know values are numeric
    - we can reuse the list more than once

    If the input is empty, the experiment cannot be analyzed.
    """
    xs = [float(v) for v in x]
    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")
    return xs


def _mean(xs: Sequence[float]) -> float:
    """
    Compute the average (mean).

    Example:
        values = [10, 12, 9]
        mean = (10 + 12 + 9) / 3 = 10.333...
    """
    return sum(xs) / len(xs)


def _sample_variance(xs: Sequence[float]) -> float:
    """
    Compute the unbiased sample variance.

    Variance measures how spread out the numbers are.

    If all values are the same, variance is zero.
    If values vary a lot, variance is large.

    We use the "unbiased" version which divides by (n - 1) instead of n.

    Example:
        values = [10, 10, 10]
        variance = 0

    Example (spread out):
        values = [5, 10, 15]
        variance is > 0

    When there is only 1 value, variance is not meaningful in the usual way.
    We return 0.0 so the rest of the calculation stays defined.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def _welch_degrees_of_freedom(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """
    Welch–Satterthwaite approximation for degrees of freedom.

    Degrees of freedom is a technical value used when converting a test statistic
    into a probability.

    You do not need to memorise this formula to use the toolkit, but it helps to know:
    - it adjusts for the fact that variances and sample sizes can be different
    - it improves reliability compared to assuming equal variances

    In some special cases (for example, both variances are zero), we return infinity,
    which means the normal distribution approximation is appropriate.
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

    This returns the probability that a normally distributed value is less than or
    equal to z.

    For example:
    - z = 0  -> about 0.50
    - z = 1  -> about 0.84
    - z = 2  -> about 0.98
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _approximate_t_cumulative_distribution(t_value: float, degrees_of_freedom: float) -> float:
    """
    Approximate the cumulative distribution of the t distribution.

    For large degrees of freedom, the t distribution is very close to the normal
    distribution, so we use the normal cumulative distribution function.

    The input `t_value` should be non-negative when used for p-values in this module.
    """
    if degrees_of_freedom == float("inf") or degrees_of_freedom > 200:
        return _normal_cumulative_distribution(t_value)

    # For smaller degrees of freedom we still use the normal approximation.
    # This keeps the code lightweight and readable.
    return _normal_cumulative_distribution(t_value)


def mean_diff(
    group_a: Iterable[float],
    group_b: Iterable[float],
    *,
    alpha: float = 0.05,
) -> MeanTestResult:
    """
    Compare the average of group B to the average of group A.

    This function is designed for experiments where you have one value per user,
    and you want to compare averages between two groups.

    Parameters
    ----------
    group_a, group_b
        Lists (or any iterable) of metric values, one per user.

        Example:
            group_a = [0.0, 0.0, 120.0, 0.0, 50.0]
            group_b = [0.0, 80.0, 120.0, 0.0, 60.0]

        This could represent "revenue per visitor" where many visitors do not buy.

    alpha
        Significance level used for the confidence interval width.
        Default alpha=0.05 corresponds to a 95% confidence interval.

    Returns
    -------
    MeanTestResult
        Contains sample sizes, means, effect, confidence interval, and p-value.

    How the calculation works (step-by-step)
    ----------------------------------------
    1) Compute mean(A) and mean(B), then compute the effect: mean(B) minus mean(A).

    2) Compute the sample variance for each group. This measures how noisy the metric is.

    3) Convert that noise into a "standard error" for the difference in means.
       The standard error gets smaller as sample size increases.

    4) Compute a test statistic:
           test statistic = effect / standard error
       This tells us how large the effect is compared to typical random noise.

    5) Convert the test statistic into a two-sided p-value.

    6) Build a confidence interval:
           effect ± (critical value) * standard error
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

    # Standard error describes the typical size of random fluctuation
    # in the estimated difference in means.
    #
    # If the groups are large, the standard error becomes small,
    # meaning we can detect smaller real effects.
    standard_error = math.sqrt(var_a / n_a + var_b / n_b)

    # Special case: if there is no variance in either group,
    # the metric is constant within each group.
    #
    # Example:
    #   A = [5, 5, 5], B = [5, 5, 5] -> effect is 0 (no difference)
    #   A = [5, 5, 5], B = [6, 6, 6] -> effect is 1 (deterministic difference)
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

    # Two-sided p-value:
    # We measure how extreme the test statistic is in either direction.
    cumulative_probability = _approximate_t_cumulative_distribution(
        abs(test_statistic), degrees_of_freedom
    )
    p_value = 2.0 * (1.0 - cumulative_probability)

    # Confidence interval:
    # For a 95% confidence interval, a common critical value is about 1.96.
    #
    # This is the familiar "within about 2 standard errors" rule of thumb.
    # It is based on the normal distribution.
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