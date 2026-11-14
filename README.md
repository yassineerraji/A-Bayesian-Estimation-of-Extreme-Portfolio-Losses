# Beyond Gaussian Risk: Bayesian Estimation of Extreme One-Day Losses

Yassine ERRAJI and Simon AMAR, *Bayesian Networks and Markov Models in Action*, 2026–2027.

## The project in one paragraph

How large is the one-day loss of the US stock market on its worst days, and how much of it does a Gaussian model miss? From 5,031 daily returns (2006–2025), we estimate the 99% Value-at-Risk and Expected Shortfall with a Bayesian Student-t model whose tail heaviness is learned from the data, and we compare it with a Gaussian benchmark. The posterior is computed with our own Metropolis–Hastings sampler, which is checked on exact answers and validated on 300 simulated data sets. For the probability of rare losses, we compare plain Monte Carlo with importance sampling.

## Where to start

1. **The report:** `report/report.pdf`. The main text (8 pages) answers the question. The appendices hold the derivations, a guide to the code, and the additional tables and figures.
2. **The notebooks:** in `code/`, numbered in reading order. They are saved with their outputs, so they can be read without running anything. Each one opens with the question it answers, what it produces, and where it sits in the project.
3. **Running the code:** `code/README.md` explains how to install the environment and run everything, and lists which notebook writes which table and figure.

## How the project is organised

```
report/                 the report
├── report.pdf          the compiled report
├── report.tex          its LaTeX source, with references.bib
├── figures/            every figure, written by the notebooks
└── tables/             every table, written by the notebooks
code/                   the analysis
├── 01_data.ipynb ... 06_checks.ipynb
├── tools.py            the functions the notebooks share: model, sampler, diagnostics, risk measures
├── data/               the data snapshot and the daily log-returns
├── results/            the posterior draws, passed from notebook 04 to notebooks 05 and 06
├── requirements.txt
└── README.md           how to install and run
```

The report contains no hand-made table or figure. Every table and figure is written by a notebook into `report/tables/` and `report/figures/`, and every number in the text is printed by a notebook. The only exceptions are the model diagram and the algorithm box, which are drawn in LaTeX.

## Reading the report next to the code

| Report section | Notebook | What to look for there |
|---|---|---|
| 2 Data | `01_data` | download and preparation of the returns; the tails against a Gaussian |
| 3 Model | `02_model_and_sampler` | the likelihood checked against SciPy; the prior predictive check |
| 4 Method | `02_model_and_sampler`, `03_validation` | the sampler's code; two tests with an exact answer; the step-size study |
| 5 Validation | `03_validation` | recovery on simulated data; coverage over 300 fits |
| 6.1 Posterior | `04_posterior` | the real-data fits, and checks with a second seed and a second sampler |
| 6.2–6.3 VaR, ES and rare losses | `05_risk` | the 2×2 comparison; plain Monte Carlo against importance sampling |
| 6.4–6.5 Checks and sensitivity | `06_checks` | predictive checks, VaR exceedances, other priors, the importance-sampling cross-check |

The notebooks run in this order. Each reads only the data and the files written by the notebooks before it.

## Reproducibility

Every random step uses the fixed seed 2026, and the package versions are pinned in `code/requirements.txt`. Running the six notebooks in order takes about 9 minutes and rewrites every table and figure of the report. A run from scratch in a fresh environment reproduces them byte for byte.
