import random

from abtk.ratio import ratio_diff


def test_ratio_diff_delta_no_effect():
    """
    If both groups are generated from the same process, the effect should be near 0
    and the confidence interval should usually include 0.
    """
    random.seed(0)

    n = 20000

    # Revenue per visitor:
    # numerator = revenue (often 0)
    # denominator = 1 visitor
    a_num = [120.0 if random.random() < 0.05 else 0.0 for _ in range(n)]
    b_num = [120.0 if random.random() < 0.05 else 0.0 for _ in range(n)]
    a_den = [1.0] * n
    b_den = [1.0] * n

    res = ratio_diff(a_num, a_den, b_num, b_den, method="delta")

    assert abs(res.effect) < 2.0
    assert res.ci_low <= 0.0 <= res.ci_high
    assert res.p_value > 0.01


def test_ratio_diff_delta_detects_effect():
    """
    Group B has a slightly higher chance of purchase, so revenue per visitor should increase.
    """
    random.seed(1)

    n = 30000
    p_a = 0.05
    p_b = 0.06  # small lift

    a_num = [120.0 if random.random() < p_a else 0.0 for _ in range(n)]
    b_num = [120.0 if random.random() < p_b else 0.0 for _ in range(n)]
    a_den = [1.0] * n
    b_den = [1.0] * n

    res = ratio_diff(a_num, a_den, b_num, b_den, method="delta")

    assert res.effect > 0.0
    assert res.p_value < 0.05


def test_ratio_diff_bootstrap_runs_and_returns_reasonable_bounds():
    """
    The bootstrap method should run and return a sensible confidence interval.
    We keep bootstrap_samples modest so tests remain fast.
    """
    random.seed(2)

    n = 8000
    p_a = 0.05
    p_b = 0.055

    a_num = [120.0 if random.random() < p_a else 0.0 for _ in range(n)]
    b_num = [120.0 if random.random() < p_b else 0.0 for _ in range(n)]
    a_den = [1.0] * n
    b_den = [1.0] * n

    res = ratio_diff(
        a_num, a_den, b_num, b_den, method="bootstrap", bootstrap_samples=400, seed=123
    )

    assert res.method == "bootstrap"
    assert res.ci_low <= res.effect <= res.ci_high
    assert 0.0 <= res.p_value <= 1.0
