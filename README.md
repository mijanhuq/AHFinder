# AHFinder: Apparent Horizon Finder

An implementation of the apparent horizon location algorithm from [Huq, Choptuik & Matzner (2000)](https://arxiv.org/abs/gr-qc/0002076), recreated through AI-assisted coding.

## The Experiment

This repository represents an experiment in AI-assisted scientific computing. The goal was to recreate a numerical relativity algorithm that one of the authors (M. Huq) originally developed **30 years ago** during his PhD research—accomplished in a matter of **hours** through collaboration with Claude (Anthropic's AI assistant).

<p float="left">
  <img src="doc/assets/Cartoon_Gemini_Pro_Part1.png" width="400" />
  <img src="doc/assets/Cartoon_Gemini_Pro_part2.png" width="400" /> 
</p>

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

![Flow of prompts used in this work](doc/assets/Gemini_Pro_FlowDiagram.png)

### Key Best Practices

When working with Claude there are a number of best practices one needs to follow:
* Requests should be put together in a bit-sized chunks where you and Claude are iterating. If you are getting Claude to jump right in and work, then the request should be short and even better yet with a success criteria. You review and if all good then go to the next. If not all good, then iterate with Claude till you get it right.
* Alternatively, a better practice is to ask Claude to propose a plan and you review the plan. Ask Claude to break the work down into chunks. Then once you agree then proceed. This is the approach I took in this project.
* At each prompt, have a success criteria. Better yet, ask Claude to create a test and put it into a test folder. Something you can ask Claude to rerun to validate any future changes. Just as you would with test-driven development.
* Ask Claude to create a folder structure. Use Claude.md to enforce structurions or constraints. we did that in this project.
* Create a tests/ folder and critical to ask Claude to add tests along the way.
* Just as a human developer such as you or perhaps a graduate student or a junior developer must, if something quantitative is implemented ensure that it is numerically correct. Have Claude write tests and carefully review. 
* Claude will sometime detect that tests are failing or the trends in a graph look wrong and may suggest a fix before you even see it. 
* Set the Claude context to make the LLM watch out for errors.
* Bottom line, test thoroughly. In this case, where we use finite differencing, convergence test, test, test test...
* Claude will use these tests to find bugs. In our case, we found some key bugs as a result of the tests. 
* Claude will suggest alternate approaches than what was used in the original paper (in our example). Specify if you want that. Here given we are reimplementing 26+ year old algorithms, I gave Claude some leeway and it was good in its suggestions. Always review, review, review
* Claude makes a choice of code implementation - it is not always the most generic and often directed code rather than reuseable extendable code. Watch for this and set the context to have it write reuseable code. 
* At the end of each session hav Claude save its context. In my case, I use Journal.md to save all prompts and responses. I use Claude.md to enforce this. Claude can and will lose context session to session and it can and will go down the same rabbit holes if you are not there to remind it.

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
