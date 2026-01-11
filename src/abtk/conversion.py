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

The basic model (why conversion rate is a probability)
Each user either converts or does not convert. That means each user outcome can be treated like a Bernoulli random variable:
        •	1 with probability p (they convert)
        •	0 with probability 1 − p (they do not)

The conversion rate we calculate from data is just an estimate of p:

p_hat = conversions / total_users

The key point is that p_hat will change from sample to sample even if the true probability p stays the same. The purpose of the confidence interval and p-value is to quantify how big that random variation could be.

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

    Why we use the Wilson interval
    ------------------------------
    For a conversion metric, each user produces either:
    - 1 (they converted), or
    - 0 (they did not convert).

    If we observe:
    - successes = number of conversions
    - n = total users

    then the observed conversion rate is:

        p_hat = successes / n

    where:
    - p_hat (pronounced “p hat”) is our estimate of the true conversion probability p.

    A very common "first" confidence interval is the so-called normal approximation:

        p_hat ± z * sqrt( p_hat * (1 - p_hat) / n )

    Terms in that formula:
    - p_hat: observed conversion rate (successes / n)
    - z: critical value from the standard normal distribution
        (for a 95% confidence interval, z is about 1.96)
    - sqrt(...): square root
    - p_hat * (1 - p_hat): an estimate of the variance of a Bernoulli outcome
    - n: sample size

    This interval is simple, but it often behaves poorly in exactly the cases we care about
    in product experimentation:
    - when conversion is rare (p_hat near 0),
    - when conversion is extremely common (p_hat near 1),
    - when sample sizes are small,
    - or when you observe 0 conversions (successes = 0) or 100% conversion (successes = n).

    In those edge cases, the normal approximation can:
    - give confidence bounds below 0 or above 1 (which makes no sense for a probability),
    - produce intervals that are too narrow (overconfident) when data is sparse.

    The Wilson interval is used because it stays well-behaved in these situations.
    It almost always produces more realistic uncertainty, especially when counts are small.

    Intuition (what Wilson is doing differently)
    --------------------------------------------
    Think of p as an unknown probability between 0 and 1.
    When n is small, our estimate p_hat is shaky, so we should be cautious.

    Wilson’s method can be understood as:
    - starting from the same idea of "estimate ± uncertainty",
    - but adjusting both the *center* of the interval and the *width* of the interval
    to account for small samples in a more reliable way.

    A useful mental model:
    - the normal approximation treats p_hat as if it is already a stable estimate,
    which is not true when n is small or p_hat is near 0/1.
    - Wilson pulls the interval slightly toward the middle and gives it a more realistic width.

    What the Wilson formula is computing (definitions)
    --------------------------------------------------
    Inside this function we compute:

    - p_hat = successes / n
    where:
        successes is the number of 1s (conversions)
        n is the total number of observations (users)

    - z2 = z * z
    where:
        z is the chosen critical value (about 1.96 for a 95% interval)

    Then Wilson forms an adjusted denominator:

        denom = 1 + (z^2 / n)

    This denom does two things:
    - it prevents the interval from becoming too narrow when n is small
    - it keeps the bounds inside [0, 1] in a natural way

    It also forms an adjusted center (the interval midpoint):

        center = ( p_hat + (z^2 / (2n)) ) / denom

    Terms:
    - p_hat is the observed conversion rate
    - z^2 / (2n) is a small adjustment that matters most when n is small
    - denom rescales the whole expression

    Finally it computes a margin (how far we go above and below the center):

        margin = z * sqrt( ( p_hat*(1 - p_hat) + z^2/(4n) ) / n ) / denom

    Terms:
    - p_hat*(1 - p_hat) is the usual Bernoulli variance estimate
    - z^2/(4n) is another stability adjustment for small samples
    - dividing by n reflects that estimates get more stable as sample size grows
    - dividing by denom matches the same rescaling used in the center

    The result is:

        low  = center - margin
        high = center + margin

    and we clamp them to [0, 1] because conversion rates cannot be negative or exceed 1.

    In short
    --------
    We use Wilson here because conversion data often has low rates and noisy outcomes.
    Wilson intervals behave sensibly in those cases and avoid overconfident or invalid bounds.
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
    # Confidence interval for the difference in conversion rates:

    # We first build a confidence interval for each group's conversion rate.
    # That gives:
    #   rate(A) is likely between [a_low, a_high]
    #   rate(B) is likely between [b_low, b_high]

    # Then we combine them into a difference interval:
    #   The smallest plausible (B - A) happens when B is as low as possible
    #   and A is as high as possible:  b_low - a_high

    #   The largest plausible (B - A) happens when B is as high as possible
    #   and A is as low as possible:   b_high - a_low

    # This approach is widely used because it stays stable even when conversion
    # is rare (for example, 0 conversions in a group).
    a_low, a_high = _wilson_interval(conv_a, n_a, z)
    b_low, b_high = _wilson_interval(conv_b, n_b, z)

    ci_low = b_low - a_high
    ci_high = b_high - a_low

    # Two-sided z-test with pooled proportion
    # P-value calculation:

    # For the p-value we test the idea that "A and B have the same true conversion rate".
    # If that were true, the best single estimate of that shared conversion rate is the
    # pooled rate across both groups:
    #   pooled = (conversions in A + conversions in B) / (users in A + users in B)

    # We then compare the observed difference (rate_b - rate_a) to how much random
    # variation we would expect if both groups really came from the same probability.
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
