"""
CUPED variance reduction for mean metrics.

CUPED stands for "Controlled Experiments Using Pre-Experiment Data".

The problem it addresses
------------------------
Many experiment metrics are noisy.
For example: revenue per visitor during the experiment.
Most users spend 0, a few spend a lot, and the average moves around due to randomness.

If you have a value for each user measured before the experiment started
(a "covariate"), you can use it to remove predictable variation.

A realistic example
-------------------
- metric: revenue per visitor during the experiment window
- covariate: revenue per visitor during the week before the experiment

Users who spent more before the experiment tend to spend more during the experiment,
even if the product change had no effect. That correlation is noise that CUPED reduces.

How CUPED works (intuitively)
-----------------------------
CUPED builds an adjusted metric:

    adjusted = metric - theta * (covariate - mean_covariate)

Terms:
- metric: the per-user metric measured during the experiment
- covariate: the per-user value measured before the experiment
- mean_covariate: the average covariate across all users used to compute theta
- theta: a coefficient that removes the linear relationship between metric and covariate

The key property:
- The adjusted metric has the same expected mean effect of treatment.
- The adjusted metric often has smaller variance, which tightens confidence intervals.

Estimating theta
----------------
A standard choice is:

    theta = Cov(metric, covariate) / Var(covariate)

If the covariate has no variation (Var = 0), CUPED cannot reduce variance and we set theta to 0.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .mean import mean_diff
from .types import CUPEDResult


def _to_float_list(x: Iterable[float]) -> list[float]:
    xs = [float(v) for v in x]
    if len(xs) == 0:
        raise ValueError("Input must contain at least 1 observation.")
    return xs


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sample_covariance(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length.")
    n = len(x)
    if n < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)


def _sample_variance(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = _mean(x)
    return sum((xi - mx) ** 2 for xi in x) / (n - 1)


def estimate_theta(metric: Sequence[float], covariate: Sequence[float]) -> float:
    """
    Estimate the CUPED coefficient theta.

    theta is chosen to reduce variance in the adjusted metric.
    A standard estimator is:

        theta = Cov(metric, covariate) / Var(covariate)

    If Var(covariate) is 0, theta is set to 0.
    """
    var_c = _sample_variance(covariate)
    if var_c == 0.0:
        return 0.0
    cov_mc = _sample_covariance(metric, covariate)
    return cov_mc / var_c


def cuped_adjust(
    metric: Iterable[float],
    covariate: Iterable[float],
    *,
    theta: float | None = None,
) -> Tuple[list[float], float]:
    """
    Apply CUPED adjustment to a per-user metric.

    Parameters
    ----------
    metric:
        Per-user experiment metric values.

    covariate:
        Per-user pre-experiment values (measured before treatment assignment).

    theta:
        Optional. If not provided, theta is estimated from the inputs.

    Returns
    -------
    (adjusted_metric, theta_used)

    The adjustment is:
        adjusted_i = metric_i - theta * (covariate_i - mean_covariate)
    """
    m = _to_float_list(metric)
    c = _to_float_list(covariate)
    if len(m) != len(c):
        raise ValueError("metric and covariate must have the same length.")

    mean_c = _mean(c)
    theta_used = estimate_theta(m, c) if theta is None else float(theta)

    adjusted = [mi - theta_used * (ci - mean_c) for mi, ci in zip(m, c)]
    return adjusted, theta_used


def cuped_mean_diff(
    metric_a: Iterable[float],
    metric_b: Iterable[float],
    covariate_a: Iterable[float],
    covariate_b: Iterable[float],
    *,
    alpha: float = 0.05,
) -> CUPEDResult:
    """
    Compare mean metrics using CUPED adjustment.

    This computes theta using the combined data (A and B together), then applies the same
    adjustment to both groups. That keeps the adjustment symmetric across variants.

    Returns a CUPEDResult containing theta, baseline means, adjusted means, and the
    statistical test result on the adjusted metric.
    """
    m_a = _to_float_list(metric_a)
    m_b = _to_float_list(metric_b)
    c_a = _to_float_list(covariate_a)
    c_b = _to_float_list(covariate_b)

    if len(m_a) != len(c_a):
        raise ValueError("Group A metric and covariate must have the same length.")
    if len(m_b) != len(c_b):
        raise ValueError("Group B metric and covariate must have the same length.")

    baseline_mean_a = _mean(m_a)
    baseline_mean_b = _mean(m_b)

    combined_metric = m_a + m_b
    combined_covariate = c_a + c_b
    theta = estimate_theta(combined_metric, combined_covariate)

    adj_a, _ = cuped_adjust(m_a, c_a, theta=theta)
    adj_b, _ = cuped_adjust(m_b, c_b, theta=theta)

    adjusted_mean_a = _mean(adj_a)
    adjusted_mean_b = _mean(adj_b)

    res = mean_diff(adj_a, adj_b, alpha=alpha)

    return CUPEDResult(
        theta=theta,
        baseline_mean_a=baseline_mean_a,
        baseline_mean_b=baseline_mean_b,
        adjusted_mean_a=adjusted_mean_a,
        adjusted_mean_b=adjusted_mean_b,
        effect=res.effect,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        p_value=res.p_value,
        n_a=res.n_a,
        n_b=res.n_b,
        alpha=alpha,
    )
