"""
Ratio metric A/B testing (for metrics like revenue per visitor)

A ratio metric has the form:

    ratio = (total numerator) / (total denominator)

Examples
--------
1) Revenue per visitor:
   numerator   = revenue from a visitor (often 0 for most visitors)
   denominator = 1 for each visitor
   ratio       = total revenue / total visitors

2) Revenue per booking:
   numerator   = revenue from a visitor
   denominator = number of bookings from a visitor (usually 0 or 1)
   ratio       = total revenue / total bookings

3) Bookings per visitor:
   numerator   = number of bookings from a visitor
   denominator = 1 for each visitor
   ratio       = total bookings / total visitors

Why ratio metrics need special care
-----------------------------------
A tempting approach is to compute the ratio for each user and then run a mean test.
That often goes wrong, especially when denominators can be zero (for example,
revenue per booking where most users have 0 bookings).

This module treats the ratio as a property of the group totals:
    ratio = sum(numerator) / sum(denominator)

Uncertainty estimation methods
------------------------------
This module supports two approaches:

1) Delta method ("delta"):
   - Fast.
   - Uses a well-known approximation that works well with large sample sizes.
   - Often used in experimentation platforms.

2) Bootstrap ("bootstrap"):
   - Slower but conceptually simple.
   - Re-samples users with replacement and recomputes the ratio many times.
   - The spread of those re-sampled ratios gives uncertainty.

Both return:
- effect = ratio(B) - ratio(A)
- confidence interval for effect
- p-value for "true effect is 0"
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .types import RatioTestResult


def _to_float_list(x: Iterable[float]) -> list[float]:
    xs = [float(v) for v in x]
    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")
    return xs


def _normal_cumulative_distribution(z: float) -> float:
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ratio_of_totals(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    total_den = sum(denominators)
    if total_den == 0.0:
        raise ValueError("Total denominator is 0. Ratio is not defined.")
    return sum(numerators) / total_den


def _delta_variance_for_ratio(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    """
    Delta-method variance estimate for a ratio of totals.

    We treat each user as contributing a pair (x_i, y_i) where:
      x_i is the numerator contribution
      y_i is the denominator contribution

    The group ratio is:
      r = sum(x_i) / sum(y_i)

    A practical way to estimate variance uses the quantity:
      s_i = x_i - r * y_i

    Intuition:
    - If x_i is consistently proportional to y_i, then x_i ≈ r*y_i and s_i is small.
    - The more s_i varies across users, the noisier the ratio estimate is.

    The approximate variance of r is:
      Var(r) ≈ (1 / n) * Var(s_i) / (mean(y)^2)

    This implementation uses a common equivalent form:
      Var(r) ≈ sample_variance(s_i) / (n * mean(y)^2)

    Returns a variance (not a standard deviation).
    """
    n = len(numerators)
    if n != len(denominators):
        raise ValueError("Numerators and denominators must have the same length.")
    if n < 2:
        return 0.0

    r = _ratio_of_totals(numerators, denominators)
    mean_den = sum(denominators) / n
    if mean_den == 0.0:
        raise ValueError("Mean denominator is 0. Ratio variance is not defined.")

    s = [x - r * y for x, y in zip(numerators, denominators)]
    s_mean = sum(s) / n
    sample_var_s = sum((v - s_mean) ** 2 for v in s) / (n - 1)

    return sample_var_s / (n * (mean_den**2))


def ratio_diff(
    numerators_a: Iterable[float],
    denominators_a: Iterable[float],
    numerators_b: Iterable[float],
    denominators_b: Iterable[float],
    *,
    method: str = "delta",
    bootstrap_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> RatioTestResult:
    """
    Compare ratio metrics in group B vs group A.

    Inputs
    ------
    numerators_a, denominators_a:
        Per-user numerator and denominator contributions for group A.

    numerators_b, denominators_b:
        Per-user numerator and denominator contributions for group B.

    method:
        "delta" or "bootstrap"

    bootstrap_samples:
        Number of bootstrap resamples when method="bootstrap".

    seed:
        Random seed for bootstrap reproducibility.

    alpha:
        Significance level for the confidence interval.
        alpha = 0.05 corresponds to a 95% confidence interval.

    Returns
    -------
    RatioTestResult:
        Includes ratio(A), ratio(B), effect, confidence interval, and p-value.
    """
    a_num = _to_float_list(numerators_a)
    a_den = _to_float_list(denominators_a)
    b_num = _to_float_list(numerators_b)
    b_den = _to_float_list(denominators_b)

    if len(a_num) != len(a_den):
        raise ValueError("Group A numerators and denominators must have the same length.")
    if len(b_num) != len(b_den):
        raise ValueError("Group B numerators and denominators must have the same length.")

    n_a = len(a_num)
    n_b = len(b_num)

    ratio_a = _ratio_of_totals(a_num, a_den)
    ratio_b = _ratio_of_totals(b_num, b_den)
    effect = ratio_b - ratio_a

    # Critical value for a 95% confidence interval under the normal distribution.
    # This keeps the interface consistent with the mean and conversion modules.
    critical_value = 1.96

    if method == "delta":
        var_a = _delta_variance_for_ratio(a_num, a_den)
        var_b = _delta_variance_for_ratio(b_num, b_den)
        standard_error = math.sqrt(var_a + var_b)

        if standard_error == 0.0:
            p_value = 0.0 if effect != 0.0 else 1.0
            return RatioTestResult(
                n_a=n_a,
                n_b=n_b,
                ratio_a=ratio_a,
                ratio_b=ratio_b,
                effect=effect,
                ci_low=effect,
                ci_high=effect,
                p_value=p_value,
                method="delta",
                alpha=alpha,
            )

        z_value = effect / standard_error
        p_value = 2.0 * (1.0 - _normal_cumulative_distribution(abs(z_value)))

        ci_low = effect - critical_value * standard_error
        ci_high = effect + critical_value * standard_error

        return RatioTestResult(
            n_a=n_a,
            n_b=n_b,
            ratio_a=ratio_a,
            ratio_b=ratio_b,
            effect=effect,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            method="delta",
            alpha=alpha,
        )

    if method == "bootstrap":
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100 for a stable interval.")

        rng = random.Random(seed)

        effects: list[float] = []
        for _ in range(bootstrap_samples):
            idx_a = [rng.randrange(n_a) for _ in range(n_a)]
            idx_b = [rng.randrange(n_b) for _ in range(n_b)]

            boot_a_num = [a_num[i] for i in idx_a]
            boot_a_den = [a_den[i] for i in idx_a]
            boot_b_num = [b_num[i] for i in idx_b]
            boot_b_den = [b_den[i] for i in idx_b]

            boot_ratio_a = _ratio_of_totals(boot_a_num, boot_a_den)
            boot_ratio_b = _ratio_of_totals(boot_b_num, boot_b_den)
            effects.append(boot_ratio_b - boot_ratio_a)

        effects.sort()

        # Percentile confidence interval.
        low_index = int((alpha / 2.0) * bootstrap_samples)
        high_index = int((1.0 - alpha / 2.0) * bootstrap_samples) - 1
        ci_low = effects[low_index]
        ci_high = effects[high_index]

        # Bootstrap p-value: how often the bootstrap effect is on the opposite side of 0.
        # This is a simple, commonly used heuristic.
        frac_leq_zero = sum(1 for e in effects if e <= 0.0) / bootstrap_samples
        frac_geq_zero = sum(1 for e in effects if e >= 0.0) / bootstrap_samples
        p_value = 2.0 * min(frac_leq_zero, frac_geq_zero)
        p_value = min(1.0, max(0.0, p_value))

        return RatioTestResult(
            n_a=n_a,
            n_b=n_b,
            ratio_a=ratio_a,
            ratio_b=ratio_b,
            effect=effect,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            method="bootstrap",
            alpha=alpha,
        )

    raise ValueError("method must be either 'delta' or 'bootstrap'.")
