"""
Power and sample size utilities for A/B testing.

These helpers answer two practical questions:

1) Sample size planning:
   "How many users do I need in each group to reliably detect an effect?"

2) Power estimation:
   "Given my sample size, what is the chance I detect a real effect of a certain size?"

Key terms
-------------------------
- Significance level (alpha):
  The false positive rate you are willing to accept.
  alpha = 0.05 means "about 5% false positives" under the no-effect assumption.

- Power:
  The probability you detect a real effect (when there actually is one).
  Power = 0.8 is a common target, meaning an 80% chance of detecting the effect.

- Minimal detectable effect:
  The smallest effect size you care about detecting (for example, +1 percentage point conversion).

These functions use normal approximations. That is the standard approach for planning
in large-scale online experiments, where sample sizes are typically large.

All sample sizes returned are "per group" (users in A and users in B).
"""

from __future__ import annotations

import math


def _normal_cumulative_distribution(z: float) -> float:
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_inverse_cdf(p: float) -> float:
    """
    Approximate inverse CDF for a standard normal distribution.

    This returns z such that:
        P(Z <= z) = p   where Z ~ Normal(0, 1)

    We use a well-known approximation (Acklam-style rational approximation).
    Accuracy is sufficient for power/sample size calculations.

    Input:
      p must be strictly between 0 and 1.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be between 0 and 1 (exclusive).")

    # Coefficients for approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    # Define break-points.
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def sample_size_two_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Required sample size per group for detecting a conversion-rate lift.

    Parameters
    ----------
    baseline_rate:
        Expected conversion rate in the control group (A), for example 0.08.

    minimum_detectable_effect:
        Absolute lift you want to detect, for example 0.01 means +1 percentage point.

    alpha:
        Significance level for a two-sided test.

    power:
        Desired power (probability of detecting the effect if it is real).

    Returns
    -------
    Integer sample size per group.

    Notes
    -----
    This is a standard normal-approximation formula used for planning.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("Rates must be between 0 and 1.")
    if minimum_detectable_effect == 0.0:
        raise ValueError("minimum_detectable_effect must be non-zero.")
    if power <= 0.0 or power >= 1.0:
        raise ValueError("power must be between 0 and 1 (exclusive).")

    z_alpha = _normal_inverse_cdf(1.0 - alpha / 2.0)
    z_power = _normal_inverse_cdf(power)

    p_bar = (p1 + p2) / 2.0
    numerator = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_power * math.sqrt(
        p1 * (1.0 - p1) + p2 * (1.0 - p2)
    )
    n = (numerator**2) / ((p2 - p1) ** 2)

    return int(math.ceil(n))


def power_two_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Approximate power for a two-proportion test given sample size per group.

    Returns power as a value between 0 and 1.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("Rates must be between 0 and 1.")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")

    z_alpha = _normal_inverse_cdf(1.0 - alpha / 2.0)

    standard_error_alt = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_per_group)

    # Under the alternative, the test statistic is centered at the true effect / SE.
    mean_z = (p2 - p1) / standard_error_alt

    # Two-sided rejection region: |Z| > z_alpha under the null.
    # Power = P(Z > z_alpha - mean_z) + P(Z < -z_alpha - mean_z) for Z~Normal(0,1)
    upper = 1.0 - _normal_cumulative_distribution(z_alpha - mean_z)
    lower = _normal_cumulative_distribution(-z_alpha - mean_z)
    return max(0.0, min(1.0, upper + lower))


def sample_size_two_means(
    standard_deviation: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Required sample size per group for detecting a mean difference.

    Parameters
    ----------
    standard_deviation:
        Expected standard deviation of the metric per user (in the same units as the metric).

    minimum_detectable_effect:
        Smallest mean difference you want to detect (in metric units).

    Returns
    -------
    Integer sample size per group.
    """
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if minimum_detectable_effect == 0.0:
        raise ValueError("minimum_detectable_effect must be non-zero.")
    if power <= 0.0 or power >= 1.0:
        raise ValueError("power must be between 0 and 1 (exclusive).")

    z_alpha = _normal_inverse_cdf(1.0 - alpha / 2.0)
    z_power = _normal_inverse_cdf(power)

    # For two independent groups with equal n:
    # standard error of difference in means = sqrt(2 * sigma^2 / n)
    # Solve for n.
    numerator = (z_alpha + z_power) ** 2 * 2.0 * (standard_deviation**2)
    n = numerator / (minimum_detectable_effect**2)
    return int(math.ceil(n))


def power_two_means(
    standard_deviation: float,
    minimum_detectable_effect: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Approximate power for a two-mean test given sample size per group.

    Returns power as a value between 0 and 1.
    """
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if minimum_detectable_effect == 0.0:
        raise ValueError("minimum_detectable_effect must be non-zero.")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")

    z_alpha = _normal_inverse_cdf(1.0 - alpha / 2.0)

    standard_error = math.sqrt(2.0 * (standard_deviation**2) / n_per_group)
    mean_z = minimum_detectable_effect / standard_error

    upper = 1.0 - _normal_cumulative_distribution(z_alpha - mean_z)
    lower = _normal_cumulative_distribution(-z_alpha - mean_z)
    return max(0.0, min(1.0, upper + lower))
