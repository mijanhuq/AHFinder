"""
Profile full horizon finding to understand time breakdown.
"""

import numpy as np
import time
import cProfile
import pstats
from io import StringIO
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast


def run_finder(N_s: int):
    M = 1.0
    metric = SchwarzschildMetricFast(M=M)

    finder = ApparentHorizonFinder(
        metric, N_s=N_s,
        use_sparse_jacobian=True
    )

    rho = finder.find(initial_radius=2.0, tol=1e-9, max_iter=20, verbose=False)
    return rho


if __name__ == "__main__":
    N_s = 21
    print(f"Profiling horizon finding for N_s={N_s}")
    print("="*60)

    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    rho = run_finder(N_s)

    profiler.disable()

    # Print stats sorted by cumulative time
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    print(stream.getvalue())
