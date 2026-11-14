# Beyond Gaussian Risk: code

Bayesian estimation of the one-day Value-at-Risk (VaR) and Expected Shortfall (ES) of the US stock market, with a Student-t model and our own Metropolis–Hastings sampler.

## Install

We used Python 3.12.13. The packages and their versions are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

The notebooks are saved with their outputs, so they can be read without running anything. To run them again, open them in Jupyter in the order below and run all cells, or run all of them from this folder:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Each notebook uses a fixed seed, so it reproduces the numbers of the report exactly.

| Notebook | Produces (figures and tables go to `../report/`) | Report | Running time |
|---|---|---|---|
| `01_data.ipynb` | `data/returns.csv`, `figures/data_overview.pdf`, `tables/data_summary.tex` | Section 2 | a few seconds |
| `02_model_and_sampler.ipynb` | `tables/prior_predictive.tex`, `tables/known_answers.tex`, `figures/known_answers.pdf` | Sections 3–4 | about 10 seconds |
| `03_validation.ipynb` | `tables/recovery.tex`, `figures/validation_diagnostics.pdf`, `tables/tuning.tex`, `figures/tuning.pdf`, `tables/coverage.tex`, `figures/coverage.pdf` | Sections 4–5 | **about 8 minutes** (300 fits) |
| `04_posterior.ipynb` | `results/draws_student_t.npy`, `results/draws_gaussian.npy`, `tables/posterior.tex`, `figures/posterior.pdf`, `figures/posterior_diagnostics.pdf`, `figures/running_means.pdf` | Section 6.1 | about 20 seconds |

## Files

| Path | Content |
|---|---|
| `tools.py` | the functions shared by the notebooks: figure and table helpers, the Student-t model (likelihood, priors, Jacobian), our Metropolis–Hastings sampler and its tuning, the convergence diagnostics, and VaR/ES |
| `data/` | the raw data snapshot and the daily log-returns |
| `results/` | the posterior draws of both models (after burn-in), written by notebook 04 and read by notebooks 05 and 06 |

Libraries: NumPy, SciPy, pandas and Matplotlib, plus Jupyter to run the notebooks. The sampler and all the diagnostics are our own code; no probabilistic-programming library is used.

## Data

*Fama/French 3 Factors (daily)*, from the Kenneth R. French Data Library, built from CRSP: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>.

- Snapshot "created by using the 202609 CRSP database", downloaded on 1 November 2026.
- Market return: `Mkt-RF` + `RF`, converted to daily log-returns, 3 January 2006 – 31 December 2025 (5,031 trading days).
- `01_data.ipynb` downloads the file only if `data/` does not contain it, so a later revision of the source cannot change the results.
