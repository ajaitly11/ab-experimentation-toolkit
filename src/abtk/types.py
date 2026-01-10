from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeanTestResult:
    """
    Result of comparing mean(B) - mean(A).

    All values are expressed in the same units as the input metric.
    Example: if input is revenue in £, then effect and CI are in £.
    """

    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    effect: float  # mean_b - mean_a
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float = 0.05