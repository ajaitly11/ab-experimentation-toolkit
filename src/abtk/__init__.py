"""
abtk (A/B Testing Toolkit)

A beginner-friendly experimentation toolkit with correctness-first stats.
"""

from .mean import mean_diff
from .conversion import conversion_diff
from .ratio import ratio_diff
from .health import srm_check
from .power import (
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)

__all__ = [
    "__version__",
    "mean_diff",
    "conversion_diff",
    "ratio_diff",
    "srm_check",
    "sample_size_two_proportions",
    "power_two_proportions",
    "sample_size_two_means",
    "power_two_means",
]
__version__ = "0.0.0"
