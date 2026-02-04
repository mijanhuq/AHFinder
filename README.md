# AHFinder: Apparent Horizon Finder

An implementation of the apparent horizon location algorithm from [Huq, Choptuik & Matzner (2000)](https://arxiv.org/abs/gr-qc/0002076), recreated through AI-assisted coding.

## The Experiment

This repository represents an experiment in AI-assisted scientific computing. The goal was to recreate a numerical relativity algorithm that one of the authors (M. Huq) originally developed **30 years ago** during his PhD research—accomplished in a matter of **hours** through collaboration with Claude (Anthropic's AI assistant).

### What We Built

A complete Python implementation of an apparent horizon finder for black hole spacetimes:
- Newton solver for locating surfaces where the expansion of outgoing null normals vanishes (Θ = 0)
- Cartesian finite difference stencils to avoid coordinate singularities at poles
- Support for Schwarzschild, Kerr, and Lorentz-boosted black hole metrics
- Comprehensive test suite with 27 verified tests

### The Journey (from [Journal.md](Journal.md))

| Session | What Happened |
|---------|---------------|
| **Initial Implementation** | Translated the paper's algorithm into Python. First test failed—Newton solver diverged. |
| **Debugging Round 1** | Systematic testing revealed the expansion formula used coordinate derivatives instead of covariant derivatives. Fixed by adding Christoffel symbol corrections. |
| **Performance Optimization** | Profiled the code, identified interpolation as bottleneck, implemented SciPy-based fast interpolator achieving 2-5x speedup. |
| **Documentation** | Created comprehensive test documentation with convergence graphs. |
| **Debugging Round 2** | Discovered Newton solver only converged when starting very close to the solution. Diagnosed using row-sum test: sparse Jacobian was missing critical couplings (especially to poles). Fixed by switching to dense Jacobian. Basin of attraction expanded from r₀ ∈ [1.9, 2.0] to r₀ ∈ [1.0, 3.0]. |

### Key Lesson Learned

> **Always test that mathematical identities hold.**

The bug in the sparse Jacobian would have been caught immediately by a simple test:
```python
# Jacobian row sums should equal dF/dr for uniform perturbation
row_sums = J.sum(axis=1)
dF_dr = (F(r + eps) - F(r)) / eps
assert np.allclose(row_sums, dF_dr)  # This failed! Row sums were negative!
```

This test revealed that the sparse Jacobian was dropping entries with values as large as 15.7—completely corrupting the Newton direction.

## Repository Structure

```
AHFinder/
├── src/ahfinder/           # Core algorithm
│   ├── surface.py          # Surface mesh management
│   ├── interpolation.py    # Biquartic interpolation
│   ├── stencil.py          # 27-point Cartesian stencil
│   ├── residual.py         # Expansion Θ computation
│   ├── jacobian.py         # Numerical Jacobian
│   ├── solver.py           # Newton iteration
│   ├── finder.py           # High-level API
│   └── metrics/            # Spacetime metrics
│       ├── schwarzschild.py
│       ├── kerr.py
│       └── boosted.py
├── tests/                  # Test suite (27 tests)
│   ├── test_jacobian.py    # Critical row-sum tests
│   ├── test_residual.py
│   └── ...
├── doc/
│   ├── algorithm.md        # Algorithm description
│   ├── ImplementationTests.md  # Detailed test results & graphs
│   └── graphs/             # Convergence plots
├── examples/
│   ├── find_horizon.py     # Basic usage
│   └── visualize_horizon.py
└── Journal.md              # Development narrative
```

## Quick Start

```python
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics import SchwarzschildMetric

# Find the Schwarzschild horizon (should be at r = 2M)
metric = SchwarzschildMetric(M=1.0)
finder = ApparentHorizonFinder(metric, N_s=17)
rho = finder.find(initial_radius=2.5, tol=1e-6)

print(f"Horizon radius: {finder.horizon_radius_average(rho):.6f}")
# Output: Horizon radius: 2.000290
```

## Results

The implementation successfully finds apparent horizons for:

| Spacetime | Expected | Found | Error |
|-----------|----------|-------|-------|
| Schwarzschild (M=1) | r = 2.0 | r = 2.000 | < 0.1% |
| Kerr (a=0.5) | r₊ = 1.866 | r = 1.866 | < 0.1% |
| Kerr (a=0.9) | r₊ = 1.436 | r = 1.436 | < 0.1% |

See [doc/ImplementationTests.md](doc/ImplementationTests.md) for comprehensive test results, convergence studies, and validation graphs.

## On AI-Assisted Scientific Computing

This project demonstrates that AI assistants can be effective collaborators for scientific computing:

1. **Rapid Prototyping**: The core algorithm was implemented in hours, not weeks
2. **Systematic Debugging**: AI can help trace through complex numerical issues methodically
3. **Documentation**: Comprehensive docs and tests were generated alongside the code
4. **Knowledge Transfer**: The AI could work from the original paper and implement the mathematics correctly

However, human expertise remained essential:
- Recognizing when results "looked wrong" (initial condition sensitivity)
- Guiding the debugging strategy
- Making architectural decisions
- Validating against physical intuition

The collaboration worked best as a dialogue—human insight combined with AI's ability to rapidly implement, test, and iterate.

## References

1. Huq, M.F., Choptuik, M.W., & Matzner, R.A. (2000). "Locating Boosted Kerr and Schwarzschild Apparent Horizons." Physical Review D, 66, 084024. [arXiv:gr-qc/0002076](https://arxiv.org/abs/gr-qc/0002076)

2. Thornburg, J. (1996). "Finding apparent horizons in numerical relativity." Physical Review D, 54, 4899-4918.

## License

MIT License - See LICENSE file for details.

---

*This implementation was created through collaboration between Mijan Huq and Claude (Anthropic) in February 2026.*