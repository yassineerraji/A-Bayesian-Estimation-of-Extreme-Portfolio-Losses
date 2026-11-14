# Beyond Gaussian Risk: code

Bayesian estimation of the one-day Value-at-Risk (VaR) and Expected Shortfall (ES) of the US stock market, with a Student-t model and our own Metropolis–Hastings sampler, against a Gaussian benchmark; and a comparison of plain Monte Carlo and importance sampling for the probability of rare losses.

## Read

The six notebooks are saved **with their outputs**: every table, figure and number of the report can be read in them without running anything. Each notebook opens with the question it answers, what it reads and produces, its running time, and a map of the project.

## Install

We used Python 3.12.13. The packages and their versions are in `requirements.txt`: NumPy, SciPy, pandas and Matplotlib, plus Jupyter to run the notebooks.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Open the notebooks in Jupyter in the order below and run all cells, or run all of them from this folder:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

The whole project runs in **about 9 minutes**, 7 of them for the 300 fits of `03_validation.ipynb`. Every notebook uses the fixed seed 2026, so it reproduces the numbers of the report exactly. Figures and tables are written to `../report/figures/` and `../report/tables/`, from which the report is compiled (`pdflatex report && bibtex report && pdflatex report && pdflatex report`, or `tectonic report.tex`, in `../report/`).

| Notebook | Question | Produces (in `../report/` unless stated) | Report | Running time |
|---|---|---|---|---|
| `01_data.ipynb` | What do the returns look like? | `data/returns.csv`, `figures/data_overview.pdf`, `tables/data_summary.tex` | Section 2 | a few seconds |
| `02_model_and_sampler.ipynb` | Which model, and is our sampler correct? | `tables/prior_predictive.tex`, `tables/known_answers.tex`, `figures/known_answers.pdf` | Sections 3–4 | about 10 seconds |
| `03_validation.ipynb` | Does the method recover known values, is it well tuned, are its intervals calibrated? | `tables/recovery.tex`, `figures/validation_diagnostics.pdf`, `tables/tuning.tex`, `figures/tuning.pdf`, `tables/coverage.tex`, `figures/coverage.pdf` | Sections 4–5 | **about 7.5 minutes** |
| `04_posterior.ipynb` | What is the posterior on the real data, and can we trust it? | `results/draws_student_t.npy`, `results/draws_gaussian.npy` (in `code/`), `tables/posterior.tex`, `figures/posterior.pdf`, `figures/posterior_diagnostics.pdf`, `figures/running_means.pdf` | Section 6.1 | about 20 seconds |
| `05_risk.ipynb` | What are the VaR and ES, and how should rare losses be computed? | `tables/risk.tex`, `figures/risk.pdf`, `tables/rare_event.tex`, `figures/rare_event.pdf`, `tables/rare_es.tex`, `tables/risk_mcse.tex` | Sections 6.2–6.3 | about 20 seconds |
| `06_checks.ipynb` | Does the model fit, do the priors matter, does another algorithm agree? | `tables/ppc.tex`, `figures/ppc.pdf`, `tables/exceedances.tex`, `figures/exceedances.pdf`, `tables/sensitivity.tex`, `tables/crosscheck.tex` | Sections 6.4–6.5 | about 40 seconds |

Notebooks 05 and 06 read the posterior draws saved by notebook 04.

## Files

| Path | Content |
|---|---|
| `tools.py` | the functions shared by the notebooks, written once: figure and table helpers; the Student-t model (likelihood, priors, Jacobian) and the Gaussian benchmark; our random-walk Metropolis–Hastings sampler, its tuning recipe and an independence sampler; the convergence diagnostics (split-R̂, ESS); the VaR and ES formulas and the predictive VaR and ES |
| `data/` | the raw data snapshot and the daily log-returns |
| `results/` | the posterior draws of both models (after burn-in), written by notebook 04 |
| `requirements.txt` | the package versions we used |

The samplers (Metropolis–Hastings, independence sampler, importance sampling) and all the diagnostics are our own code. NumPy, SciPy, pandas and Matplotlib serve only for random numbers, distributions, optimization, data handling and plots; no probabilistic-programming library is used.

## Data

*Fama/French 3 Factors (daily)*, from the Kenneth R. French Data Library, built from CRSP: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>.

- Snapshot "created by using the 202609 CRSP database", downloaded on 1 November 2026.
- Market return: `Mkt-RF` + `RF`, converted to daily log-returns, 3 January 2006 – 31 December 2025 (5,031 trading days).
- `01_data.ipynb` downloads the file only if `data/` does not contain it, so a later revision of the source cannot change the results.
