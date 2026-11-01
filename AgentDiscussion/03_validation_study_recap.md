# 03 · Validation study: recap

## Built

| File | Content |
|---|---|
| `src/risk.py` | `var_t`, `es_t`: closed-form Student-t VaR and ES, vectorized over draws. Checked against a brute-force Monte Carlo with 4 million draws (`assert`, within 1%) |
| `scripts/04_coverage_study.py` | 3 regimes × 100 simulated data sets → `results/coverage.csv`, a table and a figure. **7 minutes** |
| `config.toml` | + `[coverage]` and `[coverage.mcmc]` (lighter settings for this study only), + `results` in `[paths]`, + `start_spread` in `[mcmc]` |
| `src/report.py` | + a status red (`"miss"`) for intervals that miss the truth. It passes the colorblind checks next to the model blue |
| `README.md` | script 04, its runtime, `results/`, `src/risk.py` |

Outputs: `report/tables/validation_coverage.tex`, `report/figures/validation_coverage.pdf`.

## Results

Coverage of the 95% intervals, in %. A correctly calibrated method with 100 data sets lands in **90–99**.

| Regime | μ | σ | ν | VaR 99% | ES 99% | VaR 99.9% | ES 99.9% | Converged |
|---|---|---|---|---|---|---|---|---|
| ν = 2.5 | 92 | 95 | 97 | 99 | 99 | 98 | 97 | 100/100 |
| ν = 4 | 97 | 94 | 90 | 91 | 90 | 90 | 91 | 100/100 |
| ν = 8 | 95 | 94 | 94 | 97 | 96 | 95 | 94 | 100/100 |

- **All 21 coverages fall inside the calibrated range.** All 300 fits pass the convergence targets: R̂ ≤ 1.0071, ESS ≥ 1,243.
- **Standardized errors** (truth − mean)/sd average between −0.18 and +0.21, with spread 0.90–1.18. That means no material bias, and intervals of the right width.
- **ν = 4 sits at the bottom edge (90–91%).** It is one effect, not five: the VaR/ES misses fall on mostly the same data sets as the ν misses. ν is slightly overestimated (standardized error averages −0.17). Recorded in `TO_FIX`.
- **The figure** (ν = 2.5 regime): 97 of the 100 intervals for ν, and 99 of the 100 for VaR 99%, contain the truth.

## Differences from the plan: two sampler fixes found by this study

1. **Starting points: the MAP + 5 posterior sd**, no longer drawn from the prior. **This reverses decision 4 of Step 2.**
   - Why: one fit with ν = 8 data got stuck (R̂ 1.07, ESS 157). A prior draw with ν close to 2 starts on an almost flat ridge, where ν barely changes the likelihood, and drifts out too slowly. About 18% of fits would have such a chain.
   - Chains now start from points spread widely around the posterior. The notes' Ex. 6.7 uses comparable spreads.
2. **Laplace covariance from a finite-difference Hessian** (`laplace_cov` in `src/mh.py`), no longer BFGS's own estimate.
   - Why: BFGS's estimate is untouched when the search starts at the mode, which crashed the prior-only check. It can also be 70 times off, the root cause of Step 2's low-ESS fits.
   - With it, **even 1 pilot round gives an ESS of at least 5,883 on 10 of 10 data sets**. We keep 3 rounds as cheap insurance.

Minor changes:
- `sample_prior` is removed (no longer used).
- `run_chains` now builds its own starting points (new signature).
- The coverage figure's title no longer shows the unit.
- The progress print is flushed, so it shows up during long runs.

**Step 2 was rerun with these fixes.** Same data set, nearly the same numbers (μ still missed by 0.000007, as expected, since it comes from the data). All other targets PASS.

## For your review

- **Fix 1 changes a decision you approved in Step 2** (starts drawn from the prior). Please confirm.
- **Where the coverage figure goes.** I'd put the table in the main text and the figure in the appendix, because of the 8-page limit.
- **`TO_FIX`** has 3 new items: the ν = 4 edge, the lighter settings, and simulation-based calibration (SBC) as a possible extension. The "tuning robustness" item is ticked.

## Run

```bash
.venv/bin/python code/scripts/04_coverage_study.py   # about 7 minutes
```

**Next:** once you approve, the Step 4 plan: real data (fit, diagnostics, Bayesian Gaussian model).
