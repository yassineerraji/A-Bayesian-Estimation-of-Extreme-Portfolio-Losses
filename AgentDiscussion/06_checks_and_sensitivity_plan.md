# 06 · Model checks, sensitivity, cross-check: plan

(Run straight through, as you asked.)

## Goal

Everything the instructions list "beyond the requirements":
- **posterior predictive checks:** does the model reproduce the data?
- **prior sensitivity:** do the conclusions depend on the priors and on the modelling choices?
- **justification of the priors** by prior predictive simulation;
- **an independent cross-check** of the MH results with a second course method, as Ex. 6.2 of the notes does.

## Scripts

| Script | Content | Outputs |
|---|---|---|
| `07_model_checks.py` | Posterior predictive checks of both models (1,000 replicated data sets each, every one of the real size) on 6 statistics; in-sample VaR exceedances | `ppc.tex`, `ppc.pdf`, `exceedances.tex`, `numbers/checks.tex` |
| `08_sensitivity.py` | Prior predictive check of the priors; refits of the Student-t under 4 prior variants and 2 modelling variants | `prior_predictive.tex`, `sensitivity.tex`, `numbers/sensitivity.tex` |
| `09_crosscheck_is.py` | Self-normalized importance sampling with a Laplace Student-t₄ proposal (Ch. 3, Ex. 3.4): posterior means of θ and of VaR/ES, compared with MH; ESS of the weights | `crosscheck.tex`, `numbers/crosscheck.tex` |

## Decisions

1. **Predictive-check statistics, chosen to be well defined when ν < 4:**
   - worst day;
   - the 1% quantile;
   - days with a loss above 5%;
   - the sample sd;
   - asymmetry: the 1% quantile + the 99% quantile, which is 0 for a symmetric model;
   - lag-1 autocorrelation of |r|, the volatility-clustering check, **expected to fail for both models**.

   Kurtosis is excluded, since it is infinite under the model. We report the posterior predictive p-value Pr(T(y_rep) ≥ T(y) | y) and the 95% predictive interval.
2. **Exceedances:** the number of observed days whose loss beats each model's predictive VaR, against the expected (1−α)T with its binomial 95% band. In-sample, so it's a calibration check, not a forecast backtest.
3. **Sensitivity variants** (full sampler settings, about 5 s each):
   - ν prior rate 0.05 (prior mean ν = 22) and 0.5 (mean ν = 4);
   - the pitch's μ prior $\mathcal N(0, 0.01^2)$;
   - a wider σ prior, HalfNormal(0.05);
   - `nu_min = 1` (ES only needs ν > 1);
   - the 2016–2025 window (a 10-year sample).

   Reported for each: ν, predictive VaR/ES at 99% and 99.9%, and the largest R̂.
4. **Prior predictive check:** 100,000 draws of θ from the prior. Report the prior 95% ranges of the daily sd and of the 99% VaR. They should be wide but plausible: no 50% daily losses at the 99% level.
5. **Importance-sampling cross-check:** 50,000 draws from a multivariate Student-t (4 degrees of freedom) centred at the MAP, with the Laplace covariance. Weights on the log scale with log-sum-exp; ESS = 1/Σw̄².
6. **New settings in `config.toml`:** `[checks]` (number of replicates, loss threshold), `[sensitivity]` (variants and window), `[crosscheck]` (number of draws, df).

## Course flags

| Concept | Where in the notes |
|---|---|
| Posterior predictive simulation (two-stage) | Ch. 3, Ex. 3.2 |
| Self-normalized IS and the ESS of the weights | Ch. 3, Def. 3.3 |
| Laplace Student-t proposal | Ch. 3, Ex. 3.4 |
| MH against IS comparison | Ch. 6, Ex. 6.2 |
| Log-sum-exp | Ch. 1, Remark 1.3 |
