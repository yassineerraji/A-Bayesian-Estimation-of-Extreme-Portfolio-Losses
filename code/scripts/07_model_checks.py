"""Model checks: posterior predictive checks of both models, and in-sample VaR exceedances.

Posterior predictive check: replicate the whole data set (5,031 days) from the posterior many times, compute test
statistics on each replicate, and see where the observed statistics fall. Exceedances: count the observed days whose
loss beats each model's predictive VaR, against the expected count.

Reads:     config.toml; data/returns.csv; results/posterior_draws.npz
Produces:  report/tables/ppc.tex             observed statistics, 95% predictive intervals and p-values, both models
           report/figures/ppc.pdf            replicated statistics of both models against the observed ones
           report/tables/exceedances.tex     observed exceedances of the predictive VaR, against the expected count
           report/numbers/checks.tex         numbers quoted in the text
Run:       python scripts/07_model_checks.py   (about 30 s)
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
from src.risk import predictive_gauss, predictive_t  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
CHECKS, LEVELS = CFG["checks"], CFG["risk"]["levels"]
FIGURES, TABLES, RESULTS = CFG["paths"]["figures"], CFG["paths"]["tables"], CFG["paths"]["results"]
NUMBERS = TABLES.parent / "numbers"
rng_g, rng_t = np.random.default_rng(CFG["mcmc"]["seed"]).spawn(2)


def statistics(r):
    """Test statistics of return series r in % (last axis = days); every one is finite when nu > 2."""
    x = np.abs(r) - np.abs(r).mean(axis=-1, keepdims=True)  # centred |r|, for the volatility-clustering statistic
    q01, q99 = np.quantile(r, 0.01, axis=-1), np.quantile(r, 0.99, axis=-1)
    return {"Worst day (%)": r.min(axis=-1),
            "1% quantile (%)": q01,
            f"Days with a loss above {CHECKS['loss_threshold']:g}%": (r < -CHECKS["loss_threshold"]).sum(axis=-1),
            "Standard deviation (%)": r.std(axis=-1, ddof=1),
            "1% + 99% quantiles (%)": q01 + q99,
            "Autocorrelation of |r|, lag 1": (x[..., 1:] * x[..., :-1]).sum(axis=-1) / (x**2).sum(axis=-1)}


# %% 1. Load the data and the posterior draws; thin the draws to n_replicates parameter vectors per model
r_obs = 100 * load_returns(CFG).to_numpy()
saved = np.load(RESULTS / "posterior_draws.npz")
thin = lambda d: d.reshape(-1, d.shape[-1])[:: d.shape[0] * d.shape[1] // CHECKS["n_replicates"]][: CHECKS["n_replicates"]]  # noqa: E731
theta = {"Gaussian": thin(saved["gaussian"]), "Student-t": thin(saved["student_t"])}

# %% 2. Replicate the data set from each model: one parameter draw, then a whole series of the real length
# ======================================================================
# [COURSE] Posterior predictive simulation (two stages): Ch. 3, Sec. 3.1.2, Example 3.2
# Draw theta from the posterior, then data from p(y | theta): the replicates
# follow the posterior predictive distribution, parameter uncertainty included.
# ======================================================================
n, th_g, th_t = len(r_obs), 100 * theta["Gaussian"], theta["Student-t"] * [100, 100, 1]
r_rep = {"Gaussian": th_g[:, :1] + th_g[:, 1:] * rng_g.standard_normal((len(th_g), n)),
         "Student-t": th_t[:, :1] + th_t[:, 1:2] * rng_t.standard_t(th_t[:, 2:], (len(th_t), n))}
obs, rep = statistics(r_obs), {model: statistics(x) for model, x in r_rep.items()}

# %% 3. Posterior predictive p-values and 95% predictive intervals
# p = Pr(T(y_rep) >= T(y) | y): values near 0 or 1 mean the observed statistic is unusual under the model
p = {(model, name): np.mean(rep[model][name] >= obs[name]) for model in rep for name in obs}
ci = lambda x: f"[{np.quantile(x, 0.025):.3g}, {np.quantile(x, 0.975):.3g}]"  # noqa: E731
rows = [[name, f"{obs[name]:.3g}", ci(rep["Gaussian"][name]), f"{p['Gaussian', name]:.2f}",
         ci(rep["Student-t"][name]), f"{p['Student-t', name]:.2f}"] for name in obs]
write_table(["Statistic", "Observed", "95% interval", "$p$", "95% interval", "$p$"], rows, TABLES / "ppc.tex",
            top=[r" & & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Student-$t$} \\",
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"])
print("Posterior predictive checks (observed | Gaussian: 95% interval, p | Student-t: 95% interval, p)")
print("\n".join(f"  {row[0]:32s} " + "  ".join(f"{c:>16s}" for c in row[1:]) for row in rows))

# %% 4. In-sample exceedances of each model's predictive VaR
loss = -r_obs
predictive = {"Gaussian": predictive_gauss, "Student-t": predictive_t}
exc, rows = {}, []
for a in LEVELS:
    lo, hi = stats.binom.ppf([0.025, 0.975], n, 1 - a)
    for model in predictive:
        exc[model, a] = int((loss > 100 * predictive[model](saved[model.lower().replace("-", "_")], a)[0]).sum())
    rows.append([f"{100 * a:g}%", f"{(1 - a) * n:.0f}", f"{lo:.0f}--{hi:.0f}", exc["Gaussian", a], exc["Student-t", a]])
write_table(["VaR level", "Expected", "95% range", "Gaussian", "Student-$t$"], rows, TABLES / "exceedances.tex")
print("\nDays whose loss beats the predictive VaR (expected, 95% range, Gaussian, Student-t)")
print("\n".join("  " + "  ".join(f"{c!s:>10s}" for c in row) for row in rows))

# %% 5. Numbers quoted in the text
keys = ["worst", "q01", "days-loss", "sd", "asym", "acf"]
numbers = {f"ppc-{k}-obs": f"{obs[name]:.3g}" for k, name in zip(keys, obs)}
numbers |= {f"ppc-{k}-{model[0].lower()}-p": f"{p[model, name]:.2f}" for model in rep for k, name in zip(keys, obs)}
numbers |= {f"exc{100 * a:g}-{model[0].lower()}": f"{exc[model, a]}" for model, a in exc}
numbers |= {f"exc{100 * a:g}-expected": f"{(1 - a) * n:.0f}" for a in LEVELS}
worst_t = rep["Student-t"]["Worst day (%)"]  # predictive interval of the worst day under the Student-t
numbers["ppc-n-rep"] = f"{CHECKS['n_replicates']:,}"
numbers |= {"ppc-worst-t-lo": f"{-np.quantile(worst_t, 0.025):.1f}", "ppc-worst-t-hi": f"{-np.quantile(worst_t, 0.975):.1f}"}
write_numbers(numbers, NUMBERS / "checks.tex")

# %% 6. Figure: replicated statistics of both models against the observed value
set_style()
fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6))
for ax, name in zip(axes.ravel(), obs):
    both = np.concatenate([rep["Gaussian"][name], rep["Student-t"][name], [obs[name]]])
    bins = np.linspace(*np.quantile(both, [0.01, 0.99]), 50)  # the most extreme 1% on each side are not drawn
    for model in rep:
        ax.hist(rep[model][name], bins=bins, color=COLORS["gaussian" if model == "Gaussian" else "student_t"],
                alpha=0.8, label=model)
    ax.axvline(obs[name], color="#0b0b0b", lw=1.2, label="Observed")
    ax.set(xlabel=name, yticks=[])
axes[0, 0].legend(loc="upper left")
axes[0, 0].set_ylabel("Replicated data sets")
axes[1, 0].set_ylabel("Replicated data sets")
save_figure(fig, FIGURES / "ppc.pdf")
print("\n->", TABLES / "ppc.tex,", FIGURES / "ppc.pdf,", TABLES / "exceedances.tex,", NUMBERS / "checks.tex")
