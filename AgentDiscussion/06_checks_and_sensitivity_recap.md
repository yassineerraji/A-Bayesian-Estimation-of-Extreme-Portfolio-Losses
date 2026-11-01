# 06 · Model checks, sensitivity, cross-check: recap

## Built

| Script | Outputs | Time |
|---|---|---|
| `07_model_checks.py` | `ppc.tex`, `ppc.pdf`, `exceedances.tex`, `numbers/checks.tex` | 3 s |
| `08_sensitivity.py` | `prior_predictive.tex`, `sensitivity.tex`, `numbers/sensitivity.tex` | 30 s |
| `09_crosscheck_is.py` | `crosscheck.tex`, `numbers/crosscheck.tex` | 4 s |

Also:
- `sample_prior` is back in `src/model.py`, for the prior predictive check.
- `config.toml` has `[checks]`, `[sensitivity]` and `[crosscheck]`.
- Scripts 02–05 now write their quoted numbers to `report/numbers/*.tex`. The data table drops skewness and kurtosis, since both moments are infinite when ν < 3.

## Results

**Posterior predictive checks** (p = Pr(T_rep ≥ T_obs)):

| Statistic | Observed | Gaussian p | Student-t p |
|---|---|---|---|
| Worst day (%) | −12.8 | 1.00 ✗ | 0.10 |
| 1% quantile (%) | −3.6 | 1.00 ✗ | 0.55 |
| Days with a loss > 5% | 21 | 0.00 ✗ (expects 0–1) | 0.69 |
| Sd (%) | 1.25 | 0.51 | 0.96 |
| 1% + 99% quantiles | −0.375 | 1.00 ✗ | 0.96 (mild: the data are asymmetric) |
| Autocorrelation of \|r\| | 0.305 | 0.00 ✗ | 0.00 ✗ (volatility clustering) |

The Student-t's 95% predictive interval for the worst day is [−79.7%, −10.4%]: the power tail with ν ≈ 2.5 over-extrapolates.

**Exceedances of the predictive VaR** (5,031 days):

| Level | Expected (95% range) | Gaussian | Student-t |
|---|---|---|---|
| 97.5% | 126 (105–148) | 156 ✗ | 161 ✗ |
| 99% | 50 (37–65) | 105 ✗ | **51** |
| 99.9% | 5 (1–10) | 46 ✗ | 2 (conservative) |

**Prior predictive check:** before seeing data, the priors put the daily sd within [0.08%, 6.7%] and the 99% VaR within [0.19%, 15.6%]. Wide, but no absurd values.

**Sensitivity:**
- The 5 prior and bound variants (ν rate 0.05 and 0.5, the pitch's μ prior, a wider σ prior, ν > 1) all give ν = 2.47–2.48, and every risk measure within about 1% of the baseline. **The priors do not matter.**
- **The data window does:** on 2016–2025, ν = 2.67, the 99% VaR is 3.26% (against 3.60%) and the 99.9% ES is 13.4% (against 16.2%).
- All refits PASS.

**Importance-sampling cross-check:** 50,000 draws, weight ESS 40,925 (82%). Every posterior mean (μ, σ, ν, VaR and ES at 99% and 99.9%) agrees with MH within 1.5 combined Monte Carlo errors.

## What it means for the answer

The Student-t is **calibrated at 99%**: its VaR matches history and has 51 of 50 expected exceedances. The Gaussian badly underestimates every tail measure. Three caveats for the Discussion:
- at 99.9% the Student-t is **conservative** (it overshoots history);
- at 97.5% **both models are beaten too often**;
- **volatility clustering is missed by both**, the main limitation of the iid model.
