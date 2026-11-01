# 01 · Setup and data: recap

## Built

| File | Role |
|---|---|
| `.venv/` (repo root, git-ignored) | Python 3.12.13 + numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, matplotlib 3.11.2 |
| `code/requirements.txt` | those four versions, pinned |
| `code/config.toml` | `[data]`: URL, raw file name, window. `[paths]`: data and report outputs |
| `code/src/config.py` | `load_config()`: reads the TOML and turns paths into absolute paths, so scripts run from any folder |
| `code/src/data.py` | `load_returns()`: the returns as a dated Series |
| `code/src/report.py` | shared figure style, a fixed color per role, `save_figure()`, `write_table()` (LaTeX booktabs) |
| `code/scripts/01_download_data.py` | downloads the zip once, then builds `data/returns.csv` |
| `code/scripts/02_describe_data.py` | summary table + overview figure for the Data section |
| `code/README.md` | setup, run order, which script produces what, data citation and snapshot |
| `report/figures/data_overview.pdf`, `report/tables/data_summary.tex` | outputs, to be pulled into the report |

## Differences from the plan

1. **Added `src/report.py`.** Every later script makes figures and tables, so the style and the table writer are written once here.
2. **Added `raw_file` to the config**, because the file name inside the zip is tied to the URL.
3. **The plan's extremes were simple returns.** As log-returns, the worst day is **−12.78%** (from −12.0%) and the best is **+10.76%** (from +11.4%).
4. **Added `__pycache__/` to `.gitignore`.**

## What the data show (printed by `02_describe_data.py`)

5,031 days (3 Jan 2006 – 31 Dec 2025): mean 0.042%, sd 1.246%, skewness −0.51, **excess kurtosis 11.58**. **36 days beyond 4 sd, against 0.32 expected under a Gaussian.** The figure shows both the heavy tails (panel b) and the volatility clustering (panel a: 2008, 2020, 2025).

## Checks done

- Both scripts run from an unrelated folder.
- A second run of `01` reuses the snapshot (file untouched).
- The figure was rendered and inspected; the legend overlap is fixed.
- The two colors (blue and orange) pass the colorblind checks on a white page.

## For your review

- **Colors are fixed per role for every future figure:** data in gray, Student-t in blue, Gaussian in orange.
- **Figures are sized at the report's text width (16 cm)**, so the 8–9 pt fonts print at their true size.
- **Tables contain only the `tabular`.** Captions are written in the report.
- **Added to `TO_FIX`:** Python ≥ 3.11 is required (`tomllib`), and `__pycache__/` must be kept out of the zip.

## Run

```bash
cd code
../.venv/bin/python scripts/01_download_data.py
../.venv/bin/python scripts/02_describe_data.py
```

**Next:** once you approve, I write `02_…_plan.md` for Step 2 (model, MH sampler, diagnostics, first fit on simulated data).
