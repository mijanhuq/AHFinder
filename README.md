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
- Comprehensive test suite with 32 verified tests



### The Journey (from [Journal.md](Journal.md))

| Session | What Happened |
|---------|---------------|
| **Initial Implementation** | Translated the paper's algorithm into Python. First test failed—Newton solver diverged. |
| **Debugging Round 1** | Systematic testing revealed the expansion formula used coordinate derivatives instead of covariant derivatives. Fixed by adding Christoffel symbol corrections. |
| **Performance Optimization** | Profiled the code, identified interpolation as bottleneck, implemented SciPy-based fast interpolator achieving 2-5x speedup. |
| **Documentation** | Created comprehensive test documentation with convergence graphs. |
| **Debugging Round 2** | Discovered Newton solver only converged when starting very close to the solution. Diagnosed using row-sum test: sparse Jacobian was missing critical couplings (especially to poles). Fixed by switching to dense Jacobian. Basin of attraction expanded from r₀ ∈ [1.9, 2.0] to r₀ ∈ [1.0, 3.0]. |
| **Kerr & Boosted Metrics** | Extended to Kerr black holes. Found extrinsic curvature sign error by comparing Kerr(a=0) with Schwarzschild. For boosted metrics, discovered the black hole is non-stationary in the lab frame—fixed by adding ∂_t γ_ij term to extrinsic curvature. Area invariance restored (ratio 0.9999). |

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
* Bottom line, test thoroughly. In this case, where we use finite differencing, convergence test, test, test test... See [doc/ImplementationTests.md](doc/ImplementationTests.md).
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
├── tests/                  # Test suite (32 tests)
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
| Kerr (a=0.5) | r₊ = 1.866 | r = 1.897 | 1.7% |
| Kerr (a=0.7) | r₊ = 1.714 | r = 1.750 | 2.1% |
| Boosted Schwarzschild (v=0.3) | Lorentz contracted | ✓ | Area ratio: 0.9999 |
| Boosted Kerr (a=0.5, v=0.3) | Lorentz contracted | ✓ | Area ratio: 0.9996 |

**Key validations:**
- Area invariance under Lorentz boosts confirmed (ratio ≈ 1.0)
- Lorentz contraction observed (x/y ratio ≈ 0.95 for v=0.3)
- All metrics converge in 3-7 Newton iterations

See [doc/ImplementationTests.md](doc/ImplementationTests.md) for comprehensive test results, convergence studies, and validation graphs.

## On AI-Assisted Scientific Computing

This project demonstrates capabilities beyond simple code generation. Here's what effective human-AI collaboration looks like for computational physics:

### What Claude Brought to the Table

1. **Rapid Implementation from Literature**: Translated a 25-year-old paper into working Python code, handling the mathematical notation, index conventions, and numerical subtleties.

2. **Hypothesis-Driven Debugging**: When the boosted metric gave 13% wrong area, Claude systematically tested:
   - Coordinate transformations ✓
   - Null vector properties ✓
   - 4-metric determinant ✓
   - Finally identified: ∂_t γ_ij ≠ 0 (the black hole is *moving*!)

3. **Cross-Validation Strategies**: Used Schwarzschild (analytical) to validate Kerr (numerical), unboosted to validate boosted—catching a sign error that would have been nearly impossible to find by inspection.

4. **Physical Intuition in Code**: Recognized that "area should be Lorentz invariant" and used this as a debugging constraint, not just a post-hoc check. Note though that the original paper demonstrated this for Apparent Horizons. So this sentence generated by Claude picks up "Universal knowledge" from Claudes own training. This is an interesting concept because if you want to think outside the box (not in this case) then that could bias you. Something to keep in mind depending on your goals.

5. **Profiling and Optimization**: Identified that `extrinsic_curvature()` took 75% of metric computation time, proposed and implemented vectorized alternatives.

### What the Human Brought

- **Recognizing "wrongness"**: Knowing that Newton should converge from r₀=2.5, that Kerr(a=0) must match Schwarzschild, that boosted area can't be 13% larger
- **Reviewing proposed tests and iterating with Claude for completeness** 
- **Strategic direction**: When to dig deeper vs. try a different approach
- **Domain validation**: Is this physically reasonable? Does it match intuition from 30 years of experience?
- **Architectural decisions**: What abstractions make sense? What's the right API?

### The Collaboration Pattern

The most effective pattern was **iterative refinement with physics constraints**:

```
Human: "The area is 13% too large. That violates Lorentz invariance."
Claude: [Systematically tests components, finds ∂_t γ_ij ≠ 0]
Claude: "The boosted BH is moving—the metric isn't stationary.
        K_ij needs the time derivative term."
Human: "That makes sense physically. Implement and verify."
Claude: [Fixes, tests] "Area ratio now 0.9999."
```

This is genuine collaboration: human provides the "should be" from physics intuition, AI provides the "why isn't it" through systematic investigation, together we reach "now it is."

### Lessons for AI-Assisted Physics

1. **Tests are checkpoints, not just validation** - Each test in [ImplementationTests.md](doc/ImplementationTests.md) caught a specific bug. Without them, we'd still be debugging.

2. **Mathematical identities are debugging tools** - "Jacobian row sums should equal dF/dr" caught the sparse Jacobian bug instantly.

3. **AI can hold more context than you expect** - Tracking index conventions, sign conventions, and coordinate systems across a 300-line derivation.

4. **The hard bugs are physics bugs, not code bugs** - Missing covariant derivatives, wrong extrinsic curvature signs, non-stationary metrics. These require understanding the physics to even recognize as bugs.

5. **Iteration speed matters** - Going from "that's wrong" to "here's the fix" in minutes rather than days changes what's possible.

## References

1. Huq, M.F., Choptuik, M.W., & Matzner, R.A. (2000). "Locating Boosted Kerr and Schwarzschild Apparent Horizons." Physical Review D, 66, 084024. [arXiv:gr-qc/0002076](https://arxiv.org/abs/gr-qc/0002076)

2. Thornburg, J. (1996). "Finding apparent horizons in numerical relativity." Physical Review D, 54, 4899-4918.

## License

MIT License - See LICENSE file for details.

---

*This implementation was created through collaboration between Mijan Huq and Claude (Anthropic) in February 2026.*
