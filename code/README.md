# Beyond Gaussian Risk: code

Bayesian estimation of the one-day Value-at-Risk and Expected Shortfall of the US stock market, with a Student-t model and our own Metropolis–Hastings sampler.

## Setup

Requires Python ≥ 3.11 (developed with Python 3.12.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the scripts in order. They work from any folder, and each one finishes in under 10 seconds.

```bash
python scripts/01_download_data.py
python scripts/02_describe_data.py
python scripts/03_validate_simulated.py
```

## Settings

Every user-editable setting is in `config.toml`: data source and window, output paths, priors, sampler settings (seed, chains, iterations, burn-in, pilot runs), diagnostic targets, and the true values used for validation.

## Which script produces what

| Script | Produces | Used in the report |
|---|---|---|
| `scripts/01_download_data.py` | `data/raw/F-F_Research_Data_Factors_daily.csv` (raw snapshot), `data/returns.csv` | — |
| `scripts/02_describe_data.py` | `../report/figures/data_overview.pdf`, `../report/tables/data_summary.tex` | Section 2 (Data) |
| `scripts/03_validate_simulated.py` | `../report/tables/validation_recovery.tex`, `../report/figures/validation_diagnostics.pdf`, `../report/tables/prior_check.tex` | Section 5 (Validation); prior check in the appendix |

## Code structure

| Path | Content |
|---|---|
| `config.toml` | settings |
| `src/config.py` | `load_config()`: reads the settings |
| `src/data.py` | `load_returns()`: reads the processed returns |
| `src/model.py` | Student-t log-likelihood, prior (with Jacobian), log-posterior, parameter transforms, prior draws |
| `src/mh.py` | random-walk Metropolis–Hastings and the tuning recipe (MAP, pilot runs, several chains) |
| `src/diagnostics.py` | autocorrelation, split-R̂, ESS, MCSE, posterior summaries, PASS/FAIL targets |
| `src/report.py` | figure style, colors, diagnostics figure, figure and LaTeX-table writers |
| `scripts/` | one script per step, numbered in run order |

## Data

*Fama/French 3 Factors (daily)*, from the Kenneth R. French Data Library, built from CRSP.

- Snapshot: "created by using the 202609 CRSP database", downloaded on 1 November 2026.
- Market return: `Mkt-RF` + `RF`, converted to daily log-returns, 3 Jan 2006 – 31 Dec 2025 (5,031 trading days).
- The download script reuses `data/raw/` when the file is already there, so a later revision of the source cannot change the results.
