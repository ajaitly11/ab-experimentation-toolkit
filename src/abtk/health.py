"""
Experiment health checks.

A/B test analysis assumes that assignment to variants is random and unbiased.
Before looking at any metric results, it is good practice to run a few checks
to confirm the experiment data looks healthy.

This module implements Sample Ratio Mismatch (SRM) detection.

What is SRM?
------------
In most experiments, you intend to split traffic in a known ratio,
for example:
- 50% control, 50% treatment
- 90% control, 10% treatment

If the observed counts deviate too far from what you expect, it may indicate
a problem with randomisation or instrumentation.

SRM is not about "effectiveness" of a feature.
It is about whether the experiment setup is trustworthy enough to interpret.
"""

from __future__ import annotations

import math
from typing import Tuple

from .types import SRMResult


def _normal_cumulative_distribution(z: float) -> float:
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _chi_square_survival_function_df1(x: float) -> float:

    if x < 0:
        return 1.0
    z = math.sqrt(x)
    return 2.0 * (1.0 - _normal_cumulative_distribution(z))


def srm_check(
    count_a: int,
    count_b: int,
    *,
    expected_split: Tuple[float, float] = (0.5, 0.5),
    alpha: float = 0.001,
) -> SRMResult:
    """
    Survival function (1 - cumulative distribution) for a chi-square distribution with 1 degree of freedom.

    Why we need a survival function
    -------------------------------
    In the SRM check we compute a chi-square statistic that measures how far the observed
    variant counts are from the expected counts. The p-value is the probability of seeing
    a chi-square statistic at least this large *if the experiment assignment was correct*

    That is a "right tail" probability:

        p_value = P(ChiSquare >= observed_value)

    Many libraries expose this as a survival function, which is simply:

        survival(x) = 1 - CDF(x)

    We use the survival function because it directly gives the p-value for a chi-square test.

    Why 1 degree of freedom for two variants
    ----------------------------------------
    Degrees of freedom is a way of describing how many independent pieces of information
    are left after you account for constraints.

    In a two-variant experiment you have two counts (A and B), but they are not fully
    independent because the total number of users is fixed:

        count_a + count_b = total

    Once you know count_a, count_b is determined automatically (it must be total - count_a).
    That means there is effectively only 1 independent "direction" the data can vary in,
    so the chi-square test has 1 degree of freedom.

    In general:
    - For k categories with a fixed total, the degrees of freedom is (k - 1).
    - Here k = 2 (A and B), so degrees of freedom = 1.

    Helpful identity (how we compute it without a statistics library)
    -----------------------------------------------------------------
    When degrees of freedom is 1, the chi-square distribution has a convenient relationship
    to the standard normal distribution:

        If Z ~ Normal(0, 1), then Z^2 ~ ChiSquare(df=1)

    This means:

        P(ChiSquare(df=1) >= x) = P(Z^2 >= x)
                               = P(|Z| >= sqrt(x))
                               = 2 * P(Z >= sqrt(x))

    So we can compute the chi-square right tail probability using the normal cumulative
    distribution function.

    Simple example
    --------------
    Suppose the chi-square statistic is x = 3.84. Then:

        sqrt(x) ≈ 1.96

    The probability that a standard normal variable is >= 1.96 is about 0.025.
    Doubling it gives:

        p_value ≈ 2 * 0.025 = 0.05

    This matches a well-known reference point: for df=1, a chi-square value around 3.84
    corresponds roughly to a 0.05 p-value (a common significance threshold).

    """
    if count_a < 0 or count_b < 0:
        raise ValueError("Counts must be non-negative.")
    total = count_a + count_b
    if total == 0:
        raise ValueError("Total count must be > 0.")

    p_a, p_b = expected_split
    if p_a <= 0.0 or p_b <= 0.0:
        raise ValueError("Expected split probabilities must be positive.")
    if not math.isclose(p_a + p_b, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Expected split must sum to 1.0.")

    expected_a = total * p_a
    expected_b = total * p_b

    # Chi-square statistic for 2 categories:
    #   sum((observed - expected)^2 / expected)
    chi2 = ((count_a - expected_a) ** 2) / expected_a + ((count_b - expected_b) ** 2) / expected_b

    # p-value from chi-square distribution with 1 degree of freedom
    p_value = _chi_square_survival_function_df1(chi2)

    return SRMResult(
        count_a=count_a,
        count_b=count_b,
        expected_a=expected_a,
        expected_b=expected_b,
        chi2=chi2,
        p_value=p_value,
        alpha=alpha,
    )
