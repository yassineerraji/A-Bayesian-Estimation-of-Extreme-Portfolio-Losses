"""Describe the daily log-returns for the report's Data section.

Reads:     data/returns.csv (written by 01_download_data.py)
Produces:  report/figures/data_overview.pdf   (a) returns over time, (b) distribution vs a Gaussian, log scale
           report/tables/data_summary.tex     summary statistics
           report/numbers/data.tex            numbers quoted in the text
Run:       python scripts/02_describe_data.py
"""

# Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.report import COLORS, WIDTH, save_figure, set_style, write_numbers, write_table  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
FIGURES, TABLES = CFG["paths"]["figures"], CFG["paths"]["tables"]


# %% 1. Load the returns, in percent for readable outputs
r = 100 * load_returns(CFG)
m, s = r.mean(), r.std()

# %% 2. Summary statistics
# DECISION: no skewness or kurtosis: the fitted model has nu < 3, so both moments are infinite and the sample
# values estimate nothing stable. Tail counts are reported instead.
# Gaussian benchmark: number of days beyond 4 sd expected if returns were Gaussian with the same mean and sd
rows = [
    ["Trading days", f"{len(r):,}"],
    ["Period", f"{r.index[0]:%d %b %Y} -- {r.index[-1]:%d %b %Y}"],
    ["Mean (%)", f"{m:.3f}"],
    ["Standard deviation (%)", f"{s:.3f}"],
    ["Worst day (%)", f"{r.min():.2f}"],
    ["Best day (%)", f"{r.max():.2f}"],
    ["Days beyond 4 sd, observed", f"{(abs(r - m) > 4 * s).sum()}"],
    ["Days beyond 4 sd, expected if Gaussian", f"{len(r) * 2 * stats.norm.sf(4):.2f}"],
]
write_table(["Statistic", "Value"], rows, TABLES / "data_summary.tex")
print("\n".join(f"{name:40s} {value}" for name, value in rows))
keys = ["n-days", "period", "mean", "sd", "worst", "best", "beyond4-obs", "beyond4-gauss"]
write_numbers({f"data-{k}": value for k, (_, value) in zip(keys, rows)} | {"data-start-year": r.index[0].year,
              "data-end-year": r.index[-1].year, "data-worst-date": f"{r.idxmin():%d %B %Y}",
              "data-worst-loss": f"{-r.min():.2f}"},
              TABLES.parent / "numbers" / "data.tex")

# %% 3. Overview figure
set_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
# (a) returns over time: crises and volatility clustering
ax1.plot(r.index, r, color=COLORS["data"], lw=0.4)
ax1.set(title="(a) Daily log-returns", xlabel="Date", ylabel="Log-return (%)")
# (b) histogram against a Gaussian with the same mean and sd; the log scale makes the tails visible
x = np.linspace(r.min(), r.max(), 500)
ax2.hist(r, bins=100, density=True, histtype="stepfilled", color=COLORS["data"], alpha=0.35, label="Data")
ax2.plot(x, stats.norm.pdf(x, m, s), color=COLORS["gaussian"], label="Gaussian, same mean and sd")
ax2.set(title="(b) Distribution, log scale", xlabel="Log-return (%)", ylabel="Density", yscale="log", ylim=(5e-4, 5))
ax2.legend(loc="upper left")
save_figure(fig, FIGURES / "data_overview.pdf")
print("->", FIGURES / "data_overview.pdf", "and", TABLES / "data_summary.tex")
