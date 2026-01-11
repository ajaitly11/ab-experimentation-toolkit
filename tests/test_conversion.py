import random

from abtk.conversion import conversion_diff


def test_conversion_diff_no_effect_large_n():
    """
    If both groups have the same true conversion rate, the estimated effect should be
    close to 0 and the confidence interval should usually include 0.

    We use a fixed seed so the test is stable.
    """
    random.seed(0)

    true_rate = 0.08  # 8% conversion is a realistic order of magnitude
    n = 20000

    a = [1 if random.random() < true_rate else 0 for _ in range(n)]
    b = [1 if random.random() < true_rate else 0 for _ in range(n)]

    res = conversion_diff(a, b)

    assert abs(res.effect) < 0.01
    assert res.ci_low <= 0.0 <= res.ci_high
    assert res.p_value > 0.01


def test_conversion_diff_detects_effect():
    """
    If group B truly has a higher conversion rate, we should detect a positive effect.
    """
    random.seed(1)

    n = 30000
    rate_a = 0.08
    rate_b = 0.09  # +1 percentage point lift

    a = [1 if random.random() < rate_a else 0 for _ in range(n)]
    b = [1 if random.random() < rate_b else 0 for _ in range(n)]

    res = conversion_diff(a, b)

    assert res.effect > 0.005
    assert res.ci_low > 0.0
    assert res.p_value < 0.01


def test_conversion_handles_all_zero_or_all_one():
    """
    Edge cases: sometimes you get very rare conversions in small tests.

    This test checks the function does not crash and returns sensible bounds.
    """
    a = [0] * 1000
    b = [0] * 1000

    res = conversion_diff(a, b)
    assert res.rate_a == 0.0
    assert res.rate_b == 0.0
    assert res.effect == 0.0
    assert 0.0 <= res.p_value <= 1.0

    a2 = [1] * 1000
    b2 = [1] * 1000

    res2 = conversion_diff(a2, b2)
    assert res2.rate_a == 1.0
    assert res2.rate_b == 1.0
    assert res2.effect == 0.0
    assert 0.0 <= res2.p_value <= 1.0
