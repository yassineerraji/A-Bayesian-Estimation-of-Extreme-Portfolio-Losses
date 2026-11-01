"""Validate the model and the sampler on simulated data with known parameters.

Two checks:
  1. Prior only: with no data, the sampler must reproduce the known prior. This tests the prior and Jacobian
     code, which the data would otherwise swamp.
  2. Recovery: simulate a data set of the real size from known parameters, fit it, and check that the
     95% intervals contain the true values and that the diagnostics reach their targets.

Reads:     config.toml; data/returns.csv (only its length, so the simulated data have the real size)
Produces:  report/tables/prior_check.tex            prior quantiles, MH against exact (appendix)
           report/tables/validation_recovery.tex    true values, posterior summaries, diagnostics
           report/figures/validation_diagnostics.pdf
Run:       python scripts/03_validate_simulated.py   (about 20 s)
"""

# Imports
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.diagnostics import check_targets, summarize  # noqa: E402
from src.mh import run_chains  # noqa: E402
from src.model import log_likelihood, log_posterior, log_prior, sample_prior, to_natural, to_phi  # noqa: E402
from src.report import plot_diagnostics, save_figure, write_table  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
PRIOR, MCMC, TARGETS, TRUE = CFG["prior"], CFG["mcmc"], CFG["diagnostics"], CFG["validation"]
FIGURES, TABLES = CFG["paths"]["figures"], CFG["paths"]["tables"]
NAMES, LABELS = ["mu", "sigma", "nu"], [r"$\mu$", r"$\sigma$", r"$\nu$"]
# one independent random stream per section, so changing one section never changes the others
rng_prior, rng_data, rng_fit = np.random.default_rng(MCMC["seed"]).spawn(3)


# %% 1. Prior-only check: sample the prior with MH and compare its quantiles with the exact ones
# DECISION: dispersed starting points are drawn from the prior, in this check and in every fit.
starts = to_phi(sample_prior(MCMC["chains"], PRIOR, rng_prior), PRIOR["nu_min"])
chains, _ = run_chains(lambda phi: log_prior(phi, PRIOR), starts[0], starts, MCMC, rng_prior)
draws = to_natural(chains[:, MCMC["burn_in"]:], PRIOR["nu_min"])
exact = [stats.norm(0, PRIOR["mu_sd"]), stats.halfnorm(scale=PRIOR["sigma_scale"]),
         stats.expon(loc=PRIOR["nu_min"], scale=1 / PRIOR["nu_rate"])]
rows = [[label, f"{p:.0%}", f"{np.quantile(draws[..., i], p):.4g}", f"{exact[i].ppf(p):.4g}"]
        for i, label in enumerate(LABELS) for p in (0.05, 0.5, 0.95)]
write_table(["Parameter", "Quantile", "MH", "Exact"], rows, TABLES / "prior_check.tex")
print("Prior-only check (quantile: MH vs exact)")
print("\n".join(f"  {name:6s} {q:>4s}: {mh:>9s} vs {ex}" for (_, q, mh, ex), name in zip(rows, np.repeat(NAMES, 3))))

# %% 2. Simulate a data set of the real size from known parameters
# DECISION: true values = rounded maximum-likelihood fit on the real data (nu = 2.47), where the real posterior lives.
n_obs = len(load_returns(CFG))
r_sim = TRUE["mu"] + TRUE["sigma"] * rng_data.standard_t(TRUE["nu"], n_obs)
# sanity check: our hand-written Student-t log-likelihood matches SciPy's
assert np.isclose(log_likelihood(TRUE["mu"], TRUE["sigma"], TRUE["nu"], r_sim),
                  stats.t.logpdf(r_sim, TRUE["nu"], TRUE["mu"], TRUE["sigma"]).sum())

# %% 3. Fit: MAP, pilot run, then one chain from each dispersed starting point
phi_init = [np.median(r_sim), np.log(r_sim.std()), 0.0]  # rough start for the MAP search: nu = nu_min + 1
starts = to_phi(sample_prior(MCMC["chains"], PRIOR, rng_fit), PRIOR["nu_min"])
chains, accept = run_chains(lambda phi: log_posterior(phi, r_sim, PRIOR), phi_init, starts, MCMC, rng_fit)
draws = to_natural(chains, PRIOR["nu_min"])  # (chains, n_iter, 3), burn-in included

# %% 4. Recovery and diagnostics, on the draws after burn-in
summaries = [summarize(draws[:, MCMC["burn_in"]:, i]) for i in range(3)]
rows = [[label, f"{TRUE[name]:.4g}", f"{s['mean']:.4g}", f"[{s['lo']:.4g}, {s['hi']:.4g}]",
         f"{s['rhat']:.4f}", f"{s['ess']:,.0f}", f"{s['mcse']:.1e}"]
        for label, name, s in zip(LABELS, NAMES, summaries)]
write_table(["Parameter", "True", "Mean", "95% interval", r"$\hat R$", "ESS", "MCSE"], rows,
            TABLES / "validation_recovery.tex")
print("\nRecovery on simulated data")
for name, row in zip(NAMES, rows):
    print(f"  {name:6s} true {row[1]:>7s}  mean {row[2]:>8s}  95% {row[3]:24s} R-hat {row[4]}  ESS {row[5]:>6s}  MCSE {row[6]}")
print("  acceptance rates:", np.round(accept, 3))
checks = check_targets(summaries, accept, TARGETS)
checks["true values inside the 95% intervals"] = all(s["lo"] <= TRUE[n] <= s["hi"] for n, s in zip(NAMES, summaries))
print("\n".join(f"  {'PASS' if ok else 'FAIL'}  {check}" for check, ok in checks.items()))

# %% 5. Diagnostics figure
fig = plot_diagnostics(draws, LABELS, MCMC["burn_in"], truth=[TRUE[name] for name in NAMES])
save_figure(fig, FIGURES / "validation_diagnostics.pdf")
print("\n->", TABLES / "prior_check.tex,", TABLES / "validation_recovery.tex,", FIGURES / "validation_diagnostics.pdf")
