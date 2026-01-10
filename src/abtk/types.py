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