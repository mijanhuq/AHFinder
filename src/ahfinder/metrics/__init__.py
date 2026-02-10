"""
Metric data interfaces for apparent horizon finding.

Provides abstract interface and concrete implementations for various
spacetime metrics in 3+1 form.
"""

from .base import Metric
from .schwarzschild import SchwarzschildMetric
from .schwarzschild_fast import SchwarzschildMetricFast
from .kerr import KerrMetric
from .kerr_fast import KerrMetricFast
from .boosted import BoostedMetric
from .boosted_fast import FastBoostedMetric, CachedBoostedMetric, fast_boost_metric
from .numerical import NumericalMetric

__all__ = [
    "Metric",
    "SchwarzschildMetric",
    "SchwarzschildMetricFast",
    "KerrMetric",
    "KerrMetricFast",
    "BoostedMetric",
    "FastBoostedMetric",
    "CachedBoostedMetric",
    "fast_boost_metric",
    "NumericalMetric",
]
