# Beyond Gaussian Risk: code

Bayesian estimation of the one-day Value-at-Risk (VaR) and Expected Shortfall (ES) of the US stock market, with a Student-t model and our own Metropolis–Hastings sampler, against a Gaussian benchmark.

## Setup

Requires Python ≥ 3.11 (developed with Python 3.12.13; package versions pinned in `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the scripts in this order, from any folder. Every random step uses the seed in `config.toml`, so the numbers of the report are reproduced exactly. Total time: about **8 minutes** (measured: 7 min 48 s), 7 of them for `04_coverage_study.py` (300 fits). Every other script takes under 30 seconds.

```bash
python scripts/01_download_data.py      # data (downloads once, then reuses the saved snapshot)
python scripts/02_describe_data.py      # Section 2: data
python scripts/03_validate_simulated.py # Section 5: validation on one simulated data set
python scripts/04_coverage_study.py     # Section 5: coverage over 300 simulated data sets (~7 min)
python scripts/05_fit_real_data.py      # Section 6.1: posterior of both models (saves the draws)
python scripts/06_risk_measures.py      # Section 6.2: VaR and ES
python scripts/07_model_checks.py       # Section 6.3: posterior predictive checks, VaR exceedances
python scripts/08_sensitivity.py        # Sections 3 and 6.4: prior predictive check, sensitivity
python scripts/09_crosscheck_is.py      # Section 6.4: importance-sampling cross-check
```

Scripts 06–09 read the draws saved by 05. The report is then compiled from `../report/report.tex`, e.g. with `tectonic report.tex` or `pdflatex report && bibtex report && pdflatex report && pdflatex report` run in `../report/`. Every table, figure and number quoted in the text is written by these scripts into `../report/tables`, `../report/figures` and `../report/numbers`.

## Which script produces what

| Script | Produces (in `../report/` unless stated) | Report |
|---|---|---|
| `01_download_data.py` | `data/raw/F-F_Research_Data_Factors_daily.csv` (raw snapshot), `data/returns.csv` | Sec. 2 |
| `02_describe_data.py` | `figures/data_overview.pdf`, `tables/data_summary.tex`, `numbers/data.tex` | Sec. 2: Fig. 1, Table 1 |
| `03_validate_simulated.py` | `tables/validation_recovery.tex`, `figures/validation_diagnostics.pdf`, `tables/prior_check.tex`, `numbers/validation.tex` | Sec. 5; prior check in App. B |
| `04_coverage_study.py` | `code/results/coverage.csv`, `tables/validation_coverage.tex`, `figures/validation_coverage.pdf`, `numbers/coverage.tex` | Sec. 5; figure in App. B |
| `05_fit_real_data.py` | `code/results/posterior_draws.npz`, `tables/real_posterior.tex`, `figures/real_posterior.pdf`, `figures/real_diagnostics.pdf`, `numbers/fit.tex` | Sec. 6.1; diagnostics in App. B |
| `06_risk_measures.py` | `tables/risk_measures.tex`, `figures/risk_measures.pdf`, `tables/risk_mcse.tex`, `numbers/risk.tex` | Sec. 6.2; MCSE in App. B |
| `07_model_checks.py` | `tables/ppc.tex`, `figures/ppc.pdf`, `tables/exceedances.tex`, `numbers/checks.tex` | Sec. 6.3 |
| `08_sensitivity.py` | `tables/prior_predictive.tex`, `tables/sensitivity.tex`, `numbers/sensitivity.tex` | Sec. 3 (prior predictive, App. B), Sec. 6.4 |
| `09_crosscheck_is.py` | `tables/crosscheck.tex`, `numbers/crosscheck.tex` | Sec. 6.4; table in App. B |

## Code structure

| Path | Content |
|---|---|
| `config.toml` | every user-editable setting: data source and window, priors, sampler, diagnostic targets, validation, coverage study, risk levels, checks, sensitivity variants, cross-check |
| `src/config.py` | `load_config()`: reads the settings |
| `src/data.py` | `load_returns()`: reads the processed returns |
| `src/model.py` | Student-t log-likelihood, prior (with Jacobian), log-posterior, parameter transforms, prior draws; Gaussian model |
| `src/mh.py` | random-walk Metropolis–Hastings (our own), Laplace covariance, tuning recipe (MAP, pilot runs, several chains from dispersed starts) |
| `src/diagnostics.py` | autocorrelation, split-R̂, ESS, MCSE, batch-means errors, posterior summaries, PASS/FAIL targets |
| `src/risk.py` | closed-form VaR and ES (Student-t and Gaussian), predictive VaR and ES (Rao–Blackwellized mixture) |
| `src/report.py` | figure style and colors, diagnostics figure, LaTeX table and number writers |
| `scripts/` | one script per step, numbered in run order |
| `results/` | saved results of long runs (coverage study, posterior draws) |

Libraries: NumPy (random numbers, arrays), SciPy (distributions, special functions, optimizer, root finding), pandas (data handling), Matplotlib (figures). The samplers (Metropolis–Hastings and importance sampling) and all the diagnostics are our own code. No probabilistic-programming library is used.

## Data

*Fama/French 3 Factors (daily)*, from the Kenneth R. French Data Library, built from CRSP (`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`).

- Snapshot: "created by using the 202609 CRSP database", downloaded on 1 November 2026.
- Market return: `Mkt-RF` + `RF`, converted to daily log-returns, 3 Jan 2006 – 31 Dec 2025 (5,031 trading days).
- `01_download_data.py` reuses `data/raw/` when the file is already there, so a later revision of the source cannot change the results.
