import random

from abtk.cuped import cuped_mean_diff


def test_cuped_reduces_noise_and_detects_effect_more_easily():
    """
    This test constructs a metric that is strongly correlated with a pre-experiment covariate.

    CUPED should shrink the uncertainty, so the adjusted confidence interval should not
    be wider than the baseline approach in typical cases.
    """
    random.seed(0)

    n = 20000

    # Covariate: pre-period "spend tendency"
    cov_a = [random.gauss(0.0, 1.0) for _ in range(n)]
    cov_b = [random.gauss(0.0, 1.0) for _ in range(n)]

    # Metric: strongly correlated with covariate + noise
    # Add a small treatment effect to group B.
    effect = 0.05
    metric_a = [0.8 * c + random.gauss(0.0, 1.0) for c in cov_a]
    metric_b = [0.8 * c + effect + random.gauss(0.0, 1.0) for c in cov_b]

    res = cuped_mean_diff(metric_a, metric_b, cov_a, cov_b)

    assert res.theta != 0.0
    assert res.effect > 0.0
    assert res.p_value < 0.05
