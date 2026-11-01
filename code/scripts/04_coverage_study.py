"""Coverage study: do our 95% intervals contain the truth about 95% of the time?

For each tail regime nu in [coverage] nus, simulate n_datasets data sets of the real size from known parameters,
fit each one with the full recipe (lighter sampler settings), and record whether each 95% interval contains
the truth, for mu, sigma, nu, and for VaR and ES at the [coverage] levels.

Reads:     config.toml; data/returns.csv (only its length, so the simulated data have the real size)
Produces:  results/coverage.csv                      one row per fit and quantity
           report/tables/validation_coverage.tex      coverage per regime and quantity, converged fits
           report/figures/validation_coverage.pdf     intervals of nu and VaR 99% in the first regime
           report/numbers/coverage.tex                numbers quoted in the text
Run:       python scripts/04_coverage_study.py       (about 7 minutes)
"""

# Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.diagnostics import check_targets, summarize  # noqa: E402
from src.mh import run_chains  # noqa: E402
from src.model import log_posterior, to_natural  # noqa: E402
from src.report import COLORS, WIDTH, save_figure, set_style, write_numbers, write_table  # noqa: E402
from src.risk import es_t, var_t  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
PRIOR, TARGETS, TRUE, COV = CFG["prior"], CFG["diagnostics"], CFG["validation"], CFG["coverage"]
MCMC = CFG["mcmc"] | COV["mcmc"]  # DECISION: lighter sampler settings for this study only (300 fits)
FIGURES, TABLES, RESULTS = CFG["paths"]["figures"], CFG["paths"]["tables"], CFG["paths"]["results"]
LABELS = {"mu": r"$\mu$", "sigma": r"$\sigma$", "nu": r"$\nu$"}


def quantities(theta, levels):
    """Named quantities computed from theta (one vector or many draws): the parameters, then VaR and ES."""
    out = {"mu": theta[..., 0], "sigma": theta[..., 1], "nu": theta[..., 2]}
    for a in levels:
        out[f"VaR {100 * a:g}%"], out[f"ES {100 * a:g}%"] = var_t(theta, a), es_t(theta, a)
    return out


# %% 1. Sanity check: the closed-form VaR and ES match a brute-force Monte Carlo estimate
theta = np.array([TRUE["mu"], TRUE["sigma"], 4.0])
loss = -(theta[0] + theta[1] * np.random.default_rng(0).standard_t(theta[2], 4_000_000))
var_mc = np.quantile(loss, 0.99)
assert np.isclose(var_t(theta, 0.99), var_mc, rtol=0.01) and np.isclose(es_t(theta, 0.99), loss[loss >= var_mc].mean(), rtol=0.01)

# %% 2. Fit every simulated data set
# DECISION: fixed true values per regime (frequentist coverage); one independent random stream per data set.
n_obs, n_sets = len(load_returns(CFG)), COV["n_datasets"]
rngs = np.random.default_rng(CFG["mcmc"]["seed"]).spawn(len(COV["nus"]) * n_sets)
records = []
for k, rng in enumerate(rngs):
    truth = np.array([TRUE["mu"], TRUE["sigma"], COV["nus"][k // n_sets]])
    r = truth[0] + truth[1] * rng.standard_t(truth[2], n_obs)
    chains, accept = run_chains(lambda phi: log_posterior(phi, r, PRIOR), [np.median(r), np.log(r.std()), 0.0], MCMC, rng)
    draws = to_natural(chains[:, MCMC["burn_in"]:], PRIOR["nu_min"])
    summaries = {name: summarize(x) for name, x in quantities(draws, COV["levels"]).items()}
    converged = all(check_targets([summaries[p] for p in LABELS], accept, TARGETS).values())
    records += [{"nu_true": truth[2], "dataset": k % n_sets, "quantity": name, "true": float(true),
                 "converged": converged, **summaries[name]} for name, true in quantities(truth, COV["levels"]).items()]
    if (k + 1) % 25 == 0:
        print(f"  {k + 1}/{len(rngs)} fits done", flush=True)
df = pd.DataFrame(records)
RESULTS.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS / "coverage.csv", index=False)

# %% 3. Coverage table
# ======================================================================
# [COURSE] Checking interval coverage by simulation: Ch. 3, Sec. 3.3.1
# Monte Carlo error of a proportion: Ch. 1, Prop. 1.2 (here as a binomial band:
# with n data sets, a calibrated 95% interval has coverage inside this band).
# ======================================================================
df["covered"] = (df["lo"] <= df["true"]) & (df["true"] <= df["hi"])
coverage = 100 * df.pivot_table(index="nu_true", columns="quantity", values="covered", aggfunc="mean")
names = list(quantities(theta, COV["levels"]))  # column order: parameters, then VaR and ES
converged = df.drop_duplicates(["nu_true", "dataset"]).groupby("nu_true")["converged"].sum()
band = stats.binom.ppf([0.025, 0.975], n_sets, 0.95) / n_sets * 100
rows = [[f"$\\nu = {nu:g}$", *(f"{coverage.loc[nu, n]:.0f}" for n in names), f"{converged[nu]} / {n_sets}"]
        for nu in COV["nus"]]
rows.append(["Calibrated range", *[f"{band[0]:.0f}--{band[1]:.0f}"] * len(names), ""])
header = ["Regime", *(LABELS.get(n, n) for n in names), "Converged fits"]
write_table(header, rows, TABLES / "validation_coverage.tex")
print(f"\nCoverage of the 95% intervals (%), calibrated range {band[0]:.0f}-{band[1]:.0f}%:")
print(coverage[names].round(0).to_string())
print("Converged fits (R-hat, ESS and acceptance targets):", converged.to_dict())

# %% 3b. Bias and width: standardized errors (truth - posterior mean) / posterior sd, mean 0 and sd 1 if calibrated
df["z"] = (df["true"] - df["mean"]) / df["sd"]
z = df.groupby(["nu_true", "quantity"])["z"].agg(["mean", "std"])
params = df[df["quantity"].isin(LABELS)]
print(f"Standardized errors: means in [{z['mean'].min():.2f}, {z['mean'].max():.2f}], sds in [{z['std'].min():.2f}, {z['std'].max():.2f}]")
write_numbers({"cov-n-sets": n_sets, "cov-n-fits": len(rngs), "cov-band-lo": f"{band[0]:.0f}", "cov-band-hi": f"{band[1]:.0f}",
               "cov-min": f"{coverage.values.min():.0f}", "cov-max": f"{coverage.values.max():.0f}",
               "cov-converged": int(converged.sum()), "cov-rhat-max": f"{params['rhat'].max():.4f}",
               "cov-ess-min": f"{params['ess'].min():,.0f}", "cov-zmean-min": f"{z['mean'].min():.2f}",
               "cov-zmean-max": f"{z['mean'].max():.2f}", "cov-zsd-min": f"{z['std'].min():.2f}",
               "cov-zsd-max": f"{z['std'].max():.2f}", "cov-nu4-z": f"{z.loc[(4.0, 'nu'), 'mean']:.2f}",
               "cov-fig-nu": int(df[(df["nu_true"] == COV["nus"][0]) & (df["quantity"] == "nu")]["covered"].sum()),
               "cov-fig-var": int(df[(df["nu_true"] == COV["nus"][0]) & (df["quantity"] == "VaR 99%")]["covered"].sum())},
              TABLES.parent / "numbers" / "coverage.tex")

# %% 4. Figure: the intervals of nu and VaR 99% in the first regime, sorted, misses highlighted
set_style()
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
for ax, (name, label, unit, scale) in zip(axes, [("nu", r"$\nu$", "", 1), ("VaR 99%", "VaR 99%", " (%)", 100)]):
    d = df[(df["nu_true"] == COV["nus"][0]) & (df["quantity"] == name)].sort_values("mean")
    colors = np.where(d["covered"], COLORS["student_t"], COLORS["miss"])
    ax.vlines(np.arange(n_sets), scale * d["lo"], scale * d["hi"], colors=colors, lw=1.2)
    ax.axhline(scale * d["true"].iloc[0], color="#0b0b0b", lw=0.8, label="True value")
    ax.plot([], [], color=COLORS["student_t"], label="Contains the truth")  # legend entries only
    ax.plot([], [], color=COLORS["miss"], label="Misses the truth")
    ax.set(title=f"{label}: {d['covered'].sum()} of {n_sets} intervals contain the truth",
           xlabel="Simulated data set (sorted by posterior mean)", ylabel=label + unit)
axes[0].legend(loc="upper left")
save_figure(fig, FIGURES / "validation_coverage.pdf")
print("->", RESULTS / "coverage.csv,", TABLES / "validation_coverage.tex,", FIGURES / "validation_coverage.pdf")
