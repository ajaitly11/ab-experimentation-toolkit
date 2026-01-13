from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeanTestResult:
    """
    A simple container for the result of comparing two group averages (means).

    This toolkit uses the common A/B testing convention:

        effect = mean(B) - mean(A)

    So:
      - if effect is positive, B is higher than A (good if "higher is better")
      - if effect is negative, B is lower than A

    Example
    -------
    Suppose:
        A = [10, 12, 9, 11, 10]   -> mean(A) = 10.4
        B = [11, 12, 13, 10, 12]  -> mean(B) = 11.6

    Then:
        effect = 11.6 - 10.4 = 1.2

    Confidence interval (CI)
    ------------------------
    The CI gives a range of values that are consistent with the data.
    A 95% CI of (0.2, 2.1) can be read as:

      "Based on the sample we observed, the true effect is likely between 0.2 and 2.1."

    p-value
    -------
    The p-value answers this question:

      "If the true effect was actually 0, how unusual would our observed result be?"

    A small p-value means the observed effect would be hard to explain by random noise alone.
    """

    # sample sizes
    n_a: int
    n_b: int

    # group averages
    mean_a: float
    mean_b: float

    # mean_b - mean_a
    effect: float

    # confidence interval for the effect
    ci_low: float
    ci_high: float

    # two-sided p-value for testing "effect == 0"
    p_value: float

    # alpha is the "significance level" used to set the CI width (default 0.05 -> 95% CI)
    alpha: float = 0.05


@dataclass(frozen=True)
class ConversionTestResult:
    """
    Result of comparing conversion rates between two groups.

    A conversion is a yes/no outcome:
      - 1 means "converted" (for example: made a booking)
      - 0 means "did not convert"

    We follow the same convention as the mean test:
        effect = rate(B) - rate(A)

    Example
    -------
    Suppose:
      Group A: 120 conversions out of 2000 users  -> rate(A) = 0.06
      Group B: 150 conversions out of 2000 users  -> rate(B) = 0.075

    Then:
      effect = 0.075 - 0.06 = 0.015

    That means B increased conversion by 1.5 percentage points.

    Notes on the confidence interval
    --------------------------------
    For conversion rates, some common formulas behave badly when rates are near 0 or 1,
    or when there are very few conversions. This toolkit uses a method based on
    Wilson intervals (a well-known approach for proportions) and combines them to
    get an interval for the difference in rates.
    """

    n_a: int
    n_b: int

    conversions_a: int
    conversions_b: int

    rate_a: float
    rate_b: float

    effect: float  # rate_b - rate_a

    ci_low: float
    ci_high: float

    p_value: float

    alpha: float = 0.05


@dataclass(frozen=True)
class RatioTestResult:
    """
    Result of comparing ratio metrics between two groups.

    A ratio metric is something like:

        ratio = (total numerator) / (total denominator)

    Real examples:
    - revenue per visitor:
        numerator   = revenue
        denominator = visitors (often 1 per visitor)
    - bookings per visitor:
        numerator   = bookings (0 or 1 per visitor, sometimes more)
        denominator = visitors (often 1 per visitor)
    - revenue per booking:
        numerator   = revenue
        denominator = bookings (0 for non-bookers, 1 for bookers)

    We compare groups using:
        effect = ratio(B) - ratio(A)

    The result includes a confidence interval and a p-value for the hypothesis that
    the true effect is 0.
    """

    n_a: int
    n_b: int

    ratio_a: float
    ratio_b: float

    effect: float

    ci_low: float
    ci_high: float

    p_value: float

    method: str  # "delta" or "bootstrap"
    alpha: float = 0.05


@dataclass(frozen=True)
class SRMResult:
    """
    Sample Ratio Mismatch (SRM) check result.

    SRM means the observed traffic split across variants does not match the intended split.

    Example:
      Intended split: 50% A, 50% B
      Observed users: A=60,000, B=40,000

    That is unusual enough that it may indicate:
    - a bug in randomisation
    - a logging issue
    - targeting rules applied incorrectly
    - bots or filtering affecting one group more than the other

    In experimentation platforms, SRM is usually a "stop and investigate" signal.
    """

    count_a: int
    count_b: int
    expected_a: float
    expected_b: float
    chi2: float
    p_value: float
    alpha: float = 0.001  # platforms often use a stricter threshold than 0.05
