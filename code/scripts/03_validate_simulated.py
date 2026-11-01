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
           report/numbers/validation.tex             numbers quoted in the text
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
from src.model import log_likelihood, log_posterior, log_prior, to_natural, to_phi  # noqa: E402
from src.report import PARAM_LABELS, PARAM_SCALE, plot_diagnostics, save_figure, write_numbers, write_table  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
PRIOR, MCMC, TARGETS, TRUE = CFG["prior"], CFG["mcmc"], CFG["diagnostics"], CFG["validation"]
FIGURES, TABLES = CFG["paths"]["figures"], CFG["paths"]["tables"]
NAMES = ["mu", "sigma", "nu"]
TRUTH = np.array([TRUE[name] for name in NAMES])
# one independent random stream per section, so changing one section never changes the others
rng_prior, rng_data, rng_fit = np.random.default_rng(MCMC["seed"]).spawn(3)


# %% 1. Prior-only check: sample the prior with MH and compare its quantiles with the exact ones
prior_center = to_phi(np.array([0, PRIOR["sigma_scale"], PRIOR["nu_min"] + 1 / PRIOR["nu_rate"]]), PRIOR["nu_min"])
chains, _ = run_chains(lambda phi: log_prior(phi, PRIOR), prior_center, MCMC, rng_prior)
draws = to_natural(chains[:, MCMC["burn_in"]:], PRIOR["nu_min"]) * PARAM_SCALE  # displayed units (%)
exact = [stats.norm(0, PRIOR["mu_sd"]), stats.halfnorm(scale=PRIOR["sigma_scale"]),
         stats.expon(loc=PRIOR["nu_min"], scale=1 / PRIOR["nu_rate"])]
rows = [[label, f"{p:.0%}", f"{np.quantile(draws[..., i], p):.4g}", f"{PARAM_SCALE[i] * exact[i].ppf(p):.4g}"]
        for i, label in enumerate(PARAM_LABELS) for p in (0.05, 0.5, 0.95)]
write_table(["Parameter", "Quantile", "MH", "Exact"], rows, TABLES / "prior_check.tex")
prior_dev = max(abs(float(mh) / float(ex) - 1) for _, _, mh, ex in rows if float(ex) != 0)  # largest relative gap
print(f"Prior-only check (quantile: MH vs exact; largest relative gap {100 * prior_dev:.1f}%)")
print("\n".join(f"  {name:6s} {q:>4s}: {mh:>9s} vs {ex}" for (_, q, mh, ex), name in zip(rows, np.repeat(NAMES, 3))))

# %% 2. Simulate a data set of the real size from known parameters
# DECISION: true values = rounded maximum-likelihood fit on the real data (nu = 2.47), where the real posterior lives.
n_obs = len(load_returns(CFG))
r_sim = TRUE["mu"] + TRUE["sigma"] * rng_data.standard_t(TRUE["nu"], n_obs)
# sanity check: our hand-written Student-t log-likelihood matches SciPy's
assert np.isclose(log_likelihood(TRUE["mu"], TRUE["sigma"], TRUE["nu"], r_sim),
                  stats.t.logpdf(r_sim, TRUE["nu"], TRUE["mu"], TRUE["sigma"]).sum())

# %% 3. Fit: MAP, pilot runs, then several chains from dispersed starting points
phi_init = [np.median(r_sim), np.log(r_sim.std()), 0.0]  # rough start for the MAP search: nu = nu_min + 1
chains, accept = run_chains(lambda phi: log_posterior(phi, r_sim, PRIOR), phi_init, MCMC, rng_fit)
draws = to_natural(chains, PRIOR["nu_min"]) * PARAM_SCALE  # (chains, n_iter, 3), burn-in included, displayed units
truth = TRUTH * PARAM_SCALE

# %% 4. Recovery and diagnostics, on the draws after burn-in
summaries = [summarize(draws[:, MCMC["burn_in"]:, i]) for i in range(3)]
rows = [[label, f"{t:.4g}", f"{s['mean']:.4g}", f"[{s['lo']:.4g}, {s['hi']:.4g}]",
         f"{s['rhat']:.4f}", f"{s['ess']:,.0f}", f"{s['mcse']:.2g}"]
        for label, t, s in zip(PARAM_LABELS, truth, summaries)]
write_table(["Parameter", "True", "Mean", "95% interval", r"$\hat R$", "ESS", "MCSE"], rows,
            TABLES / "validation_recovery.tex")
print("\nRecovery on simulated data")
for name, row in zip(NAMES, rows):
    print(f"  {name:6s} true {row[1]:>7s}  mean {row[2]:>8s}  95% {row[3]:24s} R-hat {row[4]}  ESS {row[5]:>6s}  MCSE {row[6]}")
print("  acceptance rates:", np.round(accept, 3))
checks = check_targets(summaries, accept, TARGETS)
checks["true values inside the 95% intervals"] = all(s["lo"] <= t <= s["hi"] for t, s in zip(truth, summaries))
print("\n".join(f"  {'PASS' if ok else 'FAIL'}  {check}" for check, ok in checks.items()))
# the maximum likelihood fit of the same simulated data tells whether a miss comes from the data or from the sampler
nu_mle, mu_mle, _ = stats.t.fit(r_sim)
print(f"  SciPy MLE on the simulated data: mu = {100 * mu_mle:.4g}%, nu = {nu_mle:.3g}")
(mu_s, _, nu_s), (mu_t, _, nu_t) = summaries, truth
write_numbers({"val-prior-dev": f"{100 * prior_dev:.0f}", "val-mu-true": f"{mu_t:.2g}", "val-sigma-true": f"{truth[1]:.2g}", "val-mu-mean": f"{mu_s['mean']:.3f}",
               "val-mu-lo": f"{mu_s['lo']:.3f}", "val-mu-hi": f"{mu_s['hi']:.4f}", "val-mu-mle": f"{100 * mu_mle:.3f}",
               "val-nu-true": f"{nu_t:g}", "val-nu-mean": f"{nu_s['mean']:.2f}", "val-nu-lo": f"{nu_s['lo']:.2f}",
               "val-nu-hi": f"{nu_s['hi']:.2f}", "val-acc-min": f"{accept.min():.2f}", "val-acc-max": f"{accept.max():.2f}",
               "val-rhat-max": f"{max(s['rhat'] for s in summaries):.4f}",
               "val-ess-min": f"{min(s['ess'] for s in summaries):,.0f}"}, TABLES.parent / "numbers" / "validation.tex")

# %% 5. Diagnostics figure
fig = plot_diagnostics(draws, PARAM_LABELS, MCMC["burn_in"], truth=truth)
save_figure(fig, FIGURES / "validation_diagnostics.pdf")
print("\n->", TABLES / "prior_check.tex,", TABLES / "validation_recovery.tex,", FIGURES / "validation_diagnostics.pdf")
