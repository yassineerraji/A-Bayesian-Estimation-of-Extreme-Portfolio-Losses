# 01 · Setup and data: plan

## Goal

A project skeleton, a working environment, a config file, and the data ready for modelling, with the figure and table that the report's Data section needs.

## Structure (mirrors the zip)

```
BNM/
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.toml            # every user-editable setting
│   ├── src/
│   │   ├── __init__.py        # empty
│   │   ├── config.py          # load_config(): reads config.toml
│   │   └── data.py            # load_returns(): reads the processed returns
│   ├── scripts/
│   │   ├── 01_download_data.py
│   │   └── 02_describe_data.py
│   └── data/
│       ├── raw/               # downloaded file, untouched
│       └── returns.csv        # date, r
├── report/
│   ├── figures/
│   └── tables/
└── AgentDiscussion/, context/, INSTRUCTIONS.md, EXPLANATION.md, TO_FIX_BEFORE_SUBMISSION.md   (not in the zip)
```

Later steps add `src/model.py`, `src/mh.py`, `src/diagnostics.py`, `src/risk.py` and scripts `03_…` onwards.

## Decisions to review

1. **Config format: `config.toml`**, read with Python's built-in `tomllib`.
   - Why: no extra package, comments allowed, easy to read.
   - Cost: needs Python ≥ 3.11 (flagged in `TO_FIX`).
   - Alternatives: YAML (needs PyYAML), or a `config.py` (code rather than config).
2. **The config grows step by step.** It holds only what has been built so far. Step 1 adds `[data]` (URL, start and end dates) and `[paths]`.
3. **How scripts import `src/`.** Each script starts with one line that adds `code/` to Python's path, so `python scripts/01_download_data.py` works with no install step.
   - Alternative: `pip install -e .`. It's more standard, but it's an extra step that graders can miss.
4. **The "constants" section of each script = values read from `config.toml`.** This adapts the "constants" rule in `INSTRUCTIONS.md`. Hard-coded constants appear only if they are not user settings.
5. **Environment:** Python 3.12 in `.venv`, created with `uv`. Packages: numpy, scipy, pandas, matplotlib, with exact versions in `requirements.txt`.
6. **The download is skipped if `data/raw/` already holds the file.** The snapshot is never overwritten silently, so results reproduce offline even after the source is revised.

## Assumptions (will appear as `# ASSUMPTION:` in the code)

- Market return $R_t$ = `Mkt-RF` + `RF` (the Fama–French definition), given in %, so we divide by 100.
- The source rounds returns to 0.01%. That is negligible next to daily moves of about 1%.
- Log-return: $r_t = \log(1+R_t)$.
- Window 2006-01-01 → 2025-12-31 gives 5,031 days, with no missing values. Checked on the file: worst day −12.0%, best +11.4%.

## What `02_describe_data.py` produces

- **Figure** `report/figures/data_overview.pdf`, two panels:
  - (a) daily log-returns over time, showing the crises and the volatility clustering;
  - (b) a histogram on a log scale against a fitted Gaussian, showing the heavy tails.
- **Table** `report/tables/data_summary.tex`: number of days, period, mean, sd, skewness, excess kurtosis, min, max, and days beyond 4 sd (observed vs Gaussian-expected).
- **Printed** key numbers.

**Why:** the report must "point out the features of the data that the model must capture". That means the heavy tails, and the clustering that the model ignores (our main limitation).

## Not in this step

No model, no sampler, no randomness (the seed arrives in step 2). There is no course concept here, so no `[COURSE]` flags.
