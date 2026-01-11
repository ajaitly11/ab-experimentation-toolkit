"""
Conversion rate A/B testing (binary metrics)

This module helps answer questions like:

  "Did version B increase the booking conversion rate compared to version A?"

A conversion metric is binary:
- 1 means the user converted (for example, completed a booking)
- 0 means the user did not convert

You usually store one value per user:
  group_a = [0, 1, 0, 0, 1, ...]
  group_b = [0, 0, 1, 1, 0, ...]

What this module returns
------------------------
The main function, `conversion_diff(group_a, group_b)`, returns:

- conversion rate in A and B
- effect = rate(B) - rate(A)
- confidence interval for the effect
- p-value for the hypothesis "true effect is 0"

How to think about the confidence interval
------------------------------------------
A naive confidence interval for a conversion rate can behave badly when:
- conversion is rare (close to 0)
- conversion is extremely common (close to 1)
- the sample size is small

To avoid that, this module uses Wilson intervals for each group’s conversion rate,
and then combines them (Newcombe’s method) to get an interval for the difference
between the two rates.

This is a standard, practical approach in experimentation work because it stays stable
in edge cases like "0 conversions" or "100% conversions".
"""

from __future__ import annotations

import math
from typing import Iterable

from .types import ConversionTestResult


def _to_binary_list(x: Iterable[int | bool]) -> list[int]:
    """
    Convert input values into a list of 0/1 integers.

    Accepts values like:
    - 0 and 1
    - False and True

    Raises an error for anything else because conversion data should be binary.
    """
    xs: list[int] = []
    for v in x:
        if v is True:
            xs.append(1)
        elif v is False:
            xs.append(0)
        else:
            # Allow ints 0/1 explicitly
            iv = int(v)
            if iv not in (0, 1):
                raise ValueError("Conversion data must contain only 0/1 or False/True values.")
            xs.append(iv)

    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")

    return xs


def _normal_cumulative_distribution(z: float) -> float:
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    """
    Wilson score interval for a proportion.

    If p_hat = successes / n, the Wilson interval is a more stable alternative to:
      p_hat ± z * sqrt(p_hat(1-p_hat)/n)

    It behaves better when p_hat is near 0 or near 1, and when counts are small.

    Returns (low, high) bounds in [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    p_hat = successes / n
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom

    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def conversion_diff(
    group_a: Iterable[int | bool],
    group_b: Iterable[int | bool],
    *,
    alpha: float = 0.05,
) -> ConversionTestResult:
    """
    Compare conversion rates in group B vs group A.

    Inputs
    ------
    group_a, group_b:
        Per-user conversion outcomes (0/1 or False/True).

        Example:
            group_a = [0, 1, 0, 0, 0, 1]
            group_b = [0, 1, 1, 0, 1, 1]

    Returns
    -------
    ConversionTestResult:
        Includes rates, effect, confidence interval, and p-value.

    What the p-value tests
    ----------------------
    The p-value here is based on a two-sided z-test using a pooled conversion rate.

    It answers:
      "If the true conversion rate was the same in A and B, how unusual would the
       observed difference be due to random chance?"
    """
    a = _to_binary_list(group_a)
    b = _to_binary_list(group_b)

    n_a = len(a)
    n_b = len(b)

    conv_a = sum(a)
    conv_b = sum(b)

    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    effect = rate_b - rate_a

    # z critical value for a 95% confidence interval
    # (this matches what we used in the mean module)
    z = 1.96

    # Confidence interval for the difference in proportions:
    # Newcombe's method: Wilson interval for each group, then combine.
    a_low, a_high = _wilson_interval(conv_a, n_a, z)
    b_low, b_high = _wilson_interval(conv_b, n_b, z)

    ci_low = b_low - a_high
    ci_high = b_high - a_low

    # Two-sided z-test with pooled proportion
    pooled = (conv_a + conv_b) / (n_a + n_b)
    denom = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))

    if denom == 0.0:
        # This happens if pooled is 0 or 1 (everybody had the same outcome),
        # so there is no randomness left in the metric.
        p_value = 0.0 if effect != 0.0 else 1.0
    else:
        z_stat = effect / denom
        p_value = 2.0 * (1.0 - _normal_cumulative_distribution(abs(z_stat)))

    return ConversionTestResult(
        n_a=n_a,
        n_b=n_b,
        conversions_a=conv_a,
        conversions_b=conv_b,
        rate_a=rate_a,
        rate_b=rate_b,
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        alpha=alpha,
    )
