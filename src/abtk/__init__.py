"""
abtk (A/B Testing Toolkit)

A beginner-friendly experimentation toolkit with correctness-first stats.
"""

from .mean import mean_diff
from .conversion import conversion_diff
from .ratio import ratio_diff

__all__ = ["__version__", "mean_diff", "conversion_diff", "ratio_diff"]
__version__ = "0.0.0"
