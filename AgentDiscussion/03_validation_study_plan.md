# 03 · Validation study: plan

## Goal

Step 2 showed one simulated data set. One data set can't tell us whether our 95% intervals are **calibrated**: whether they contain the truth about 95% of the time. This step repeats the whole fit on many simulated data sets, in three tail regimes, and reports:
- the **coverage** of the intervals: for the parameters, and for the quantities the project is about (VaR and ES);
- how many fits pass the convergence targets. That tests the robustness of our tuning, an open item in `TO_FIX`.

**Assumed from Step 2:** we keep its data set and explain the narrow miss on μ in the report. This study is that explanation.

## New and changed files

| File | Content |
|---|---|
| `src/risk.py` | `var_t(θ, α)`, `es_t(θ, α)`: closed-form VaR and ES of the Student-t, vectorized over posterior draws. Step 5 builds on them |
| `scripts/04_coverage_study.py` | the loop over regimes × data sets; saves every fit's results, then writes the table and figure |
| `config.toml` | + `[coverage]` (regimes, number of data sets, lighter sampler settings, levels); + `results` in `[paths]` |
| `results/coverage.csv` | one row per fit: regime, data set, quantity, true value, posterior mean, 95% interval, R̂, ESS. Lets us rebuild the table without rerunning |

## Decisions to review

1. **Regimes:** ν ∈ {2.5, 4, 8}, with μ = 0.001 and σ = 0.007, and data of the real size (5,031 days).
   - ν = 2.5 is our real-data regime. ν = 4 and ν = 8 check that the method does not only work for very heavy tails.
2. **100 data sets per regime, 300 fits in total.**
   - With 100 data sets, the Monte Carlo error on a coverage near 95% is ±2.2%. An interval works correctly if its coverage falls in **about [90%, 99%]**, the 95% binomial band; the table states this band.
3. **Lighter sampler settings, for this study only:** pilot runs of 2,000 iterations (still 3 rounds), and 4 chains of 6,000 iterations with a burn-in of 1,000.
   - Measured on one data set per regime: **1.3 s per fit, ESS of about 1,800, R̂ ≤ 1.004.** That gives about **6.5 minutes** in total (stated in the README).
   - The main fits (Step 2, Step 4) keep the full settings.
4. **Quantities covered:** μ, σ, ν, **plus VaR and ES at 99% and 99.9%**, computed draw by draw from the closed-form formulas.
   - Why: the report's answer is a VaR/ES interval, so we validate that directly, not only the parameters.
   - This pulls `src/risk.py` forward from Step 5. It's only the two formulas, ~10 lines.
5. **Fixed true values per regime** (frequentist coverage), not true values redrawn from the prior for every data set (simulation-based calibration).
   - Why: easier to read ("in the ν = 2.5 world, our intervals contain the truth X% of the time"), and directly the instructions' requirement.
6. **Each data set gets its own random stream**, spawned from the seed. Any single fit can be rerun alone and gives the same result.
7. **Convergence over all 300 fits:** the table reports how many fits pass R̂ < 1.01 and ESS ≥ 1,000. If some fail, we report it rather than dropping them.

## Outputs of `04_coverage_study.py` (about 6.5 min)

- `report/tables/validation_coverage.tex`: one row per regime; columns are the coverage (%) of μ, σ, ν, VaR₉₉, ES₉₉, VaR₉₉.₉, ES₉₉.₉, plus the number of fits passing the targets.
- `report/figures/validation_coverage.pdf`: for the ν = 2.5 regime, the 100 intervals of ν and of VaR₉₉, sorted, with the true value as a line and misses highlighted. **Appendix, or main text if space allows.**
- `results/coverage.csv`.
- Printed: coverages, the binomial band, and PASS/FAIL counts.

## VaR and ES formulas

Here $q = t_\nu^{-1}(\alpha)$ and $f_\nu$ is the standard Student-t density:

$$\text{VaR}_\alpha = -\mu + \sigma q, \qquad \text{ES}_\alpha = -\mu + \sigma\,\frac{f_\nu(q)}{1-\alpha}\,\frac{\nu + q^2}{\nu - 1}.$$

An `assert` checks both against a brute-force Monte Carlo estimate.

## Course flags planned

| Concept | Where in the notes |
|---|---|
| Checking interval coverage by simulation | Ch. 3, Sec. 3.3.1 |
| Monte Carlo error of a proportion (binomial band) | Ch. 1, Prop. 1.2 |
| Posterior summaries of a function of θ (VaR, ES) | Ch. 1, Def. 1.4, eq. (1.7) |
| Plus every flag already in `src/` | — |

## Not in this step

Real data (Step 4), the Gaussian models and the 2×2 comparison (Step 5).
