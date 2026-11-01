# 04 · Real data: plan

## Goal

Fit our validated Student-t model to the 5,031 real daily returns. Show that the fit can be trusted (diagnostics), report the posterior with its uncertainty, and fit the **Bayesian Gaussian model** that Step 5's 2×2 comparison needs. The posterior draws are saved, so Steps 5 and 6 never rerun the sampler.

## New and changed files

| File | Content |
|---|---|
| `src/model.py` | + the Gaussian model: `to_natural_gauss`, `log_posterior_gauss` on φ = (μ, log s), Jacobian included |
| `scripts/05_fit_real_data.py` | fit both models, check the targets, write the table and figures, save the draws |
| `results/posterior_draws.npz` | draws after burn-in, chains kept apart: Student-t (4, 18,000, 3) and Gaussian (4, 18,000, 2) |
| `src/report.py` | `plot_diagnostics` reused as is (no truth line) |

## Decisions to review

1. **Same full sampler settings as Step 2** (`[mcmc]`: 4 chains × 20,000, burn-in 2,000, 3 pilot rounds, starts at MAP + 5 sd), and the same PASS/FAIL targets.
2. **The Bayesian Gaussian model:** $r_t \sim \mathcal N(\mu, s^2)$, with $\mu \sim \mathcal N(0, 0.001^2)$ (same as the Student-t) and $s \sim$ HalfNormal(0.02).
   - $s$ is now the standard deviation. Same prior scale as σ, so the two models differ only in their tails.
   - It is fitted with the **same** `run_chains` (d = 2). No second sampler.
3. **Quantities reported for the Student-t:** μ, σ, ν and the **implied standard deviation** $\sigma\sqrt{\nu/(\nu-2)}$. The implied sd lets us compare with the Gaussian $s$ and the sample sd (1.246%).
   - Expect a surprise: with ν ≈ 2.4 the implied sd should be well **above** the sample sd. With such heavy tails, the sample sd underestimates the spread. That matters for the 2×2 comparison (Step 5) and is worth a sentence in the report.
4. **Event probabilities:**
   - $\Pr(\nu < 4 \mid y)$: whether the 4th moment is infinite, so that kurtosis is undefined. This settles the kurtosis question in `TO_FIX`.
   - $\Pr(\nu < 3 \mid y)$: an even heavier tail.
   - The posterior 2.5% quantile of ν: how close the posterior gets to the ν > 2 bound (`TO_FIX`).
5. **Display convention: μ, σ, sd, VaR and ES in % in every report table**, so no more 0.0009953. **This also updates the two Step 2 tables** (one line each in `03_validate_simulated.py`).
6. **Sanity print, not in the report:** our posterior mean of ν against SciPy's MLE (2.47). They should be close, since the prior is weak.

## Outputs of `05_fit_real_data.py` (about 15 s)

- `report/tables/real_posterior.tex`: rows for the Student-t μ, σ, ν and sd, then the Gaussian μ and s. Columns: mean, sd, 95% interval, R̂, ESS, MCSE.
- `report/figures/real_posterior.pdf`, four panels:
  - posterior histograms of μ, σ and ν;
  - the implied sd (Student-t) against $s$ (Gaussian), with the sample sd as a line.
- `report/figures/real_diagnostics.pdf`: the 3×3 trace/autocorrelation grid for the real fit (appendix).
- Printed: acceptance rates, PASS/FAIL for both models, event probabilities, the comparison with the MLE.

## Course flags planned

| Concept | Where in the notes |
|---|---|
| Probability of an event as a posterior expectation of an indicator | Ch. 1, Def. 1.4, table after eq. (1.7) |
| Ergodic averages from the chains | Ch. 5, Thm 5.2 |
| Diagnostics | Ch. 6, Sec. 6.6 |
| Plus the flags already in `src/` | — |

## Not in this step

VaR and ES (Step 5), predictive checks and sensitivity (Step 6).
