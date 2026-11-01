# 04 · Real data: recap

## Built

| File | Content |
|---|---|
| `src/model.py` | + `to_natural_gauss`, `log_posterior_gauss`: Gaussian model on (μ, log s), Jacobian included |
| `src/report.py` | + `PARAM_LABELS` and `PARAM_SCALE`: the "% in every table and figure" convention, written once; + gridlines drawn behind the bars |
| `scripts/05_fit_real_data.py` | fits both models, checks the targets, writes the table and 2 figures, saves the draws. **About 10 s** |
| `scripts/03_validate_simulated.py` | now displays in % (same numbers: μ 0.07524% instead of 0.0007524) |
| `results/posterior_draws.npz` | Student-t (4, 18,000, 3) and Gaussian (4, 18,000, 2), after burn-in |
| `README.md` | script 05 added |

Outputs: `report/tables/real_posterior.tex`, `report/figures/real_posterior.pdf`, `report/figures/real_diagnostics.pdf` (appendix).

## Results (5,031 days, 2006–2025)

| Model | Quantity | Mean | 95% interval | R̂ | ESS |
|---|---|---|---|---|---|
| Student-t | μ (%) | 0.097 | [0.073, 0.120] | 1.0004 | 6,733 |
| | σ (%) | 0.683 | [0.656, 0.710] | 1.0006 | 5,995 |
| | ν | **2.48** | [2.27, 2.71] | 1.0006 | 5,752 |
| | Implied sd (%) | **1.58** | [1.37, 1.92] | 1.0009 | 5,200 |
| Gaussian | μ (%) | 0.041 | [0.007, 0.075] | 1.0005 | 9,205 |
| | s (%) | 1.246 | [1.222, 1.271] | 1.0012 | 8,715 |

- **Convergence:** both models PASS every target. Acceptance is 0.32 for the Student-t and 0.36 for the Gaussian.
- **The fit agrees with the maximum likelihood:** SciPy's MLE gives ν = 2.470 against our posterior mean of 2.48, and μ and σ agree too. The weak prior barely moves anything.
- **Very heavy tails:** **all 72,000 draws have ν < 4** (infinite 4th moment), and **71,997 have ν < 3**.
- **The ν > 2 bound is not binding:** the 2.5% quantile of ν is 2.274.

## What this means for the next steps

1. **Kurtosis is undefined under the model.** I recommend dropping the "excess kurtosis" row from the data table (`TO_FIX`).
2. **The implied sd is 27% above the sample sd** (1.58% against 1.25%). With ν ≈ 2.5, the sample sd underestimates the spread, so the Gaussian, fitted to that sample spread, is too narrow from the start. That will be a key part of the Step 5 story.
3. **The Student-t centre (0.097%) is more than twice the Gaussian mean (0.041%).** The centre is median-like, and negative skew pulls the mean below it. Worth one sentence in the report.

## Differences from the plan

- **The fourth panel of the posterior figure shows posterior means with 95% intervals, not histograms.** The Gaussian sd is so precise (±0.012%) that its histogram was a spike that flattened the Student-t one.
- **Gridlines now sit behind the bars** in every figure (fixed in the shared style). The Step 1 and Step 2 figures were regenerated.
- **Event probabilities are printed as counts** ("71,997 of 72,000 draws"), because "1.0000" hid the 3 draws with ν ≥ 3.

## For your review

- **The figures:** `real_posterior.pdf` for the main text, `real_diagnostics.pdf` for the appendix.
- **`TO_FIX`:** 3 new items (implied sd against sample sd, location against mean, a file of LaTeX macros for numbers quoted in the text). Two items were updated with Step 4's answers (ν near its bound, kurtosis).

## Run

```bash
.venv/bin/python code/scripts/05_fit_real_data.py
```

**Next:** once you approve, the Step 5 plan: VaR and ES (posterior and predictive, at 97.5%, 99% and 99.9%), the 2×2 Gaussian/Student-t comparison, and the Monte Carlo error on these quantities.
