# 00 · Decisions agreed before building (1 Nov 2026)

These decisions replace the open points in `EXPLANATION.md` §3.1, §6 and §9.

## Project choices

| Topic | Decision |
|---|---|
| Data | Daily **US stock market return** from the Fama–French data library (value-weighted index of all US stocks, built from CRSP) |
| Window | 2006-01-03 → 2025-12-31: **5,031 trading days**, no missing values (checked), dates fixed in the config |
| Delivery | A download script, plus the saved file shipped in the zip (licence to check, see `TO_FIX_BEFORE_SUBMISSION.md`) |
| Returns | $r_t = \log(1+R_t)$, where $R_t$ = (`Mkt-RF` + `RF`)/100. Decimals in the code, % in outputs |
| Model | $r_t \overset{iid}{\sim}$ Student-$t_\nu(\mu,\sigma)$ |
| Priors | $\mu \sim \mathcal N(0, 0.001^2)$, $\sigma \sim$ HalfNormal(0.02), $\nu-2 \sim$ Exp(**rate** 0.1) |
| Sampler | Our own random-walk MH on $(\mu, \log\sigma, \log(\nu-2))$ with the Jacobian; 4 chains; proposal covariance $(2.38^2/3)\hat\Sigma$ from a pilot run; run length set by targets: $\hat R<1.01$, ESS ≥ 1,000, acceptance 20–50% |
| Reported | Posterior of VaR(θ) and ES(θ), plus the predictive VaR and ES. Levels: 99% (headline), ES 97.5%, 99.9%. Horizon: 1 day |
| Comparison | 2×2: {Gaussian, Student-t} × {plug-in, Bayesian}. Bayesian Gaussian: $\mu\sim\mathcal N(0,0.001^2)$, sd ~ HalfNormal(0.02) |
| Validation | True values (0.0003, 0.007, 4), plus a case with ν = 8; a coverage study on about 100 simulated data sets if runtime allows |
| Predictive checks | Kurtosis, worst day, number of days beyond 4 sd, 1% quantile, autocorrelation of $r_t^2$ |
| Sensitivity | ν rate 0.05 and 0.5; the pitch's μ prior $\mathcal N(0, 0.01^2)$; window 2016–2025 |
| Cross-check | Optional: importance sampling with a Laplace proposal (Ch. 3) |

## Engineering choices

| Topic | Decision |
|---|---|
| Environment | `.venv` + pinned `requirements.txt` (course `environment.yml` not found, see `TO_FIX_BEFORE_SUBMISSION.md`) |
| Code | Python scripts: `src/` (reusable modules) + `scripts/` (numbered steps) |
| Config | One config file holding every user-editable setting (data, priors, sampler, levels, paths) |
| Outputs | Figures as PDF in `report/figures/`; tables as LaTeX in `report/tables/` (pulled in with `\input`); fixed seeds |

## Roadmap

| Step | Content | Target |
|---|---|---|
| 1 | Setup (structure, `.venv`, config) + data download and description | 2 Nov |
| 2 | Model, MH sampler, diagnostics; first fit on simulated data | 4 Nov |
| 3 | Validation study (recovery, coverage) | 6 Nov |
| 4 | Real-data fit, Gaussian fits, diagnostics | 8 Nov |
| 5 | Risk measures (VaR/ES, predictive, 2×2) | 9 Nov |
| 6 | Posterior predictive checks, sensitivity | 11 Nov |
| 7 | Report, README, final from-scratch run, zip | 15 Nov |
| — | **Submit** | 16 Nov, 23:59 |
