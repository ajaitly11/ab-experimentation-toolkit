import random

from abtk.mean import mean_diff


def test_mean_diff_no_effect_large_n():
    """
    This test checks a very common sanity condition:

      If A and B are generated from the *same* distribution,
      the estimated effect should be close to 0,
      and the confidence interval should usually include 0.

    We use a fixed random seed so the test behaves consistently
    (otherwise it could randomly fail once in a while).
    """
    random.seed(0)

    # Both groups have the same "true" mean: 0.0
    # Standard deviation is 1.0 (a typical amount of noise).
    a = [random.gauss(0.0, 1.0) for _ in range(5000)]
    b = [random.gauss(0.0, 1.0) for _ in range(5000)]

    res = mean_diff(a, b)

    # With 5000 points per group, the observed effect should be small.
    assert abs(res.effect) < 0.1

    # Under the true null, p-values are roughly uniform.
    # We don't expect an extremely tiny p-value for this particular seed.
    assert res.p_value > 0.01

    # If there is no true difference, a reasonable CI should include 0.
    assert res.ci_low <= 0.0 <= res.ci_high


def test_mean_diff_detects_effect():
    """
    This test checks that when there *is* a real difference,
    the function notices it.

    We generate:
      - A with true mean 0.0
      - B with true mean 0.3

    With a decent sample size, we expect:
      - positive estimated effect
      - confidence interval entirely above 0
      - small p-value
    """
    random.seed(1)

    a = [random.gauss(0.0, 1.0) for _ in range(2000)]
    b = [random.gauss(0.3, 1.0) for _ in range(2000)]  # B is shifted upward

    res = mean_diff(a, b)

    # Effect should clearly be positive.
    assert res.effect > 0.15

    # With this sample size, we expect a clear signal (small p-value).
    assert res.p_value < 0.01

    # CI should suggest the true effect is > 0.
    assert res.ci_low > 0.0