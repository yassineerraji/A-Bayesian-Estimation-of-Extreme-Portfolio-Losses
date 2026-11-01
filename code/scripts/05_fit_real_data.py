"""Fit the Student-t model and the Gaussian benchmark model to the real daily returns.

Reads:     config.toml; data/returns.csv
Produces:  results/posterior_draws.npz          draws after burn-in, chains kept apart, both models
           report/tables/real_posterior.tex      posterior summaries and diagnostics, both models
           report/figures/real_posterior.pdf     posteriors of mu, sigma, nu; implied sd against the Gaussian sd
           report/figures/real_diagnostics.pdf   traces and autocorrelations of the Student-t fit (appendix)
           report/numbers/fit.tex                numbers quoted in the text
Run:       python scripts/05_fit_real_data.py   (about 15 s)
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
from src.diagnostics import check_targets, summarize  # noqa: E402
from src.mh import run_chains  # noqa: E402
from src.model import log_posterior, log_posterior_gauss, to_natural, to_natural_gauss  # noqa: E402
from src.report import (COLORS, PARAM_LABELS, PARAM_SCALE, WIDTH, plot_diagnostics, save_figure,  # noqa: E402
                        set_style, write_numbers, write_table)

# Constants (read from config.toml)
CFG = load_config()
PRIOR, MCMC, TARGETS = CFG["prior"], CFG["mcmc"], CFG["diagnostics"]
FIGURES, TABLES, RESULTS = CFG["paths"]["figures"], CFG["paths"]["tables"], CFG["paths"]["results"]
rng_t, rng_g = np.random.default_rng(MCMC["seed"]).spawn(2)  # one independent stream per model


# %% 1. Load the returns
r = load_returns(CFG).to_numpy()

# %% 2. Fit the Student-t model (same recipe and settings as on simulated data)
chains, accept_t = run_chains(lambda phi: log_posterior(phi, r, PRIOR), [np.median(r), np.log(r.std()), 0.0], MCMC, rng_t)
draws_t = to_natural(chains, PRIOR["nu_min"])  # (chains, n_iter, 3), burn-in included

# %% 3. Fit the Gaussian model with the same sampler
# DECISION: same prior on mu, and HalfNormal(sigma_scale) on the sd s, so that the two models differ only in their tails.
chains, accept_g = run_chains(lambda phi: log_posterior_gauss(phi, r, PRIOR), [r.mean(), np.log(r.std())], MCMC, rng_g)
draws_g = to_natural_gauss(chains)

# %% 4. Save the draws after burn-in, for steps 5 and 6
kept_t, kept_g = draws_t[:, MCMC["burn_in"]:], draws_g[:, MCMC["burn_in"]:]
RESULTS.mkdir(parents=True, exist_ok=True)
np.savez(RESULTS / "posterior_draws.npz", student_t=kept_t, gaussian=kept_g)

# %% 5. Posterior summaries and diagnostics, in displayed units (%)
sd_t = kept_t[..., 1] * np.sqrt(kept_t[..., 2] / (kept_t[..., 2] - 2))  # implied sd of the Student-t: sigma sqrt(nu/(nu-2))
spec = [("Student-t", label, scale * kept_t[..., i]) for i, (label, scale) in enumerate(zip(PARAM_LABELS, PARAM_SCALE))]
spec += [("Student-t", "Implied sd (%)", 100 * sd_t),
         ("Gaussian", PARAM_LABELS[0], 100 * kept_g[..., 0]), ("Gaussian", "$s$ (%)", 100 * kept_g[..., 1])]
summaries = [summarize(x) for _, _, x in spec]
rows = [[model, label, f"{s['mean']:.4g}", f"{s['sd']:.2g}", f"[{s['lo']:.4g}, {s['hi']:.4g}]",
         f"{s['rhat']:.4f}", f"{s['ess']:,.0f}", f"{s['mcse']:.2g}"] for (model, label, _), s in zip(spec, summaries)]
for i in range(len(rows) - 1, 0, -1):  # print each model name only on its first row
    rows[i][0] = "" if rows[i][0] == rows[i - 1][0] else rows[i][0]
write_table(["Model", "Quantity", "Mean", "Sd", "95% interval", r"$\hat R$", "ESS", "MCSE"], rows,
            TABLES / "real_posterior.tex")
print("Posterior summaries (displayed units)")
print("\n".join("  " + "  ".join(f"{c.replace('$', '').replace(chr(92), ''):>18s}" for c in row) for row in rows))
for model, s, accept in [("Student-t", summaries[:3], accept_t), ("Gaussian", summaries[4:], accept_g)]:
    print(f"\n{model}: acceptance rates {np.round(accept, 3)}")
    print("\n".join(f"  {'PASS' if ok else 'FAIL'}  {check}" for check, ok in check_targets(s, accept, TARGETS).items()))

# %% 6. Event probabilities and sanity check
# ======================================================================
# [COURSE] Probability of an event as a posterior expectation: Ch. 1, Def. 1.4
# Pr(A | y) = E[1{theta in A} | y], estimated by the fraction of draws in A.
# ======================================================================
nu = kept_t[..., 2]
print(f"\nPr(nu < 4 | y): {(nu < 4).sum():,} of {nu.size:,} draws   (4th moment infinite: kurtosis undefined)")
print(f"Pr(nu < 3 | y): {(nu < 3).sum():,} of {nu.size:,} draws")
print(f"2.5% posterior quantile of nu = {np.quantile(nu, 0.025):.3f}   (lower bound nu_min = {PRIOR['nu_min']})")
print(f"Sample sd = {100 * r.std(ddof=1):.3f}%   Student-t implied sd: posterior mean {100 * sd_t.mean():.3f}%")
nu_mle, mu_mle, sigma_mle = stats.t.fit(r)  # sanity check only: the weak prior should barely move the posterior
print(f"SciPy MLE: mu = {100 * mu_mle:.4f}%, sigma = {100 * sigma_mle:.4f}%, nu = {nu_mle:.3f}")
mu_s, sig_s, nu_s, sd_s, gmu_s, gs_s = summaries
write_numbers({"fit-mu": f"{mu_s['mean']:.3f}", "fit-mu-lo": f"{mu_s['lo']:.3f}", "fit-mu-hi": f"{mu_s['hi']:.3f}",
               "fit-sigma": f"{sig_s['mean']:.3f}", "fit-nu": f"{nu_s['mean']:.2f}", "fit-nu-lo": f"{nu_s['lo']:.2f}",
               "fit-nu-hi": f"{nu_s['hi']:.2f}", "fit-sd": f"{sd_s['mean']:.2f}", "fit-sd-lo": f"{sd_s['lo']:.2f}",
               "fit-sd-hi": f"{sd_s['hi']:.2f}", "fit-g-mu": f"{gmu_s['mean']:.3f}", "fit-g-s": f"{gs_s['mean']:.3f}",
               "fit-sample-sd": f"{100 * r.std(ddof=1):.3f}", "fit-n-draws": f"{nu.size:,}",
               "fit-nu-below4": f"{(nu < 4).sum():,}", "fit-nu-below3": f"{(nu < 3).sum():,}",
               "fit-nu-mle": f"{nu_mle:.2f}", "fit-acc-t-min": f"{accept_t.min():.2f}", "fit-acc-t-max": f"{accept_t.max():.2f}",
               "fit-rhat-max": f"{max(s['rhat'] for s in summaries):.4f}",
               "fit-ess-min": f"{min(s['ess'] for s in summaries):,.0f}",
               "fit-sd-ratio": f"{100 * (sd_s['mean'] / (100 * r.std(ddof=1)) - 1):.0f}"}
              | {f"cfg-{k.replace('_', '-')}": f"{v:,}" for k, v in MCMC.items()}  # sampler settings quoted in the text
              | {"cfg-kept": f"{MCMC['n_iter'] - MCMC['burn_in']:,}"}
              | {f"cov-cfg-{k.replace('_', '-')}": f"{v:,}" for k, v in CFG["coverage"]["mcmc"].items()}, TABLES.parent / "numbers" / "fit.tex")

# %% 7. Figures: posteriors, and diagnostics of the Student-t fit
set_style()
fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.0))
for ax, (_, label, x) in zip(axes, spec[:3]):
    ax.hist(x.ravel(), bins=60, density=True, color=COLORS["student_t"])
    ax.set(xlabel=label, yticks=[])
# standard deviation: posterior mean and 95% interval of each model, against the sample sd
for y, (model, s) in enumerate([("Gaussian", summaries[5]), ("Student-t", summaries[3])]):
    axes[3].errorbar(s["mean"], y, xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="o", ms=4, capsize=3,
                     color=COLORS["gaussian" if model == "Gaussian" else "student_t"])
axes[3].axvline(100 * r.std(ddof=1), color=COLORS["data"], lw=1, label="Sample sd")
axes[3].set(xlabel="Standard deviation (%)", yticks=[0, 1], yticklabels=["Gaussian", "Student-t\n(implied)"], ylim=(-0.6, 1.6))
axes[3].legend(loc="lower right")
axes[0].set_ylabel("Posterior density")
save_figure(fig, FIGURES / "real_posterior.pdf")
save_figure(plot_diagnostics(draws_t * PARAM_SCALE, PARAM_LABELS, MCMC["burn_in"]), FIGURES / "real_diagnostics.pdf")
print("\n->", RESULTS / "posterior_draws.npz,", TABLES / "real_posterior.tex,",
      FIGURES / "real_posterior.pdf,", FIGURES / "real_diagnostics.pdf")
