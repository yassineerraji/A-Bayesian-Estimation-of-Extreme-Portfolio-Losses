"""Prior predictive check and sensitivity analysis of the Student-t model.

1. Prior predictive check: what do the priors imply for the daily sd and the 99% VaR, before seeing any data?
2. Sensitivity: refit the Student-t model under other reasonable priors and modelling choices, and compare nu and
   the predictive VaR and ES with the baseline fit of 05_fit_real_data.py.

Reads:     config.toml; data/returns.csv; results/posterior_draws.npz (baseline)
Produces:  report/tables/prior_predictive.tex   prior quantiles of nu, the daily sd and the 99% VaR
           report/tables/sensitivity.tex        nu and predictive VaR/ES under each variant
           report/numbers/sensitivity.tex       numbers quoted in the text
Run:       python scripts/08_sensitivity.py   (about 1 minute)
"""

# Imports
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.diagnostics import check_targets, summarize  # noqa: E402
from src.mh import run_chains  # noqa: E402
from src.model import log_posterior, sample_prior, to_natural  # noqa: E402
from src.report import write_numbers, write_table  # noqa: E402
from src.risk import predictive_t, var_t  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
PRIOR, MCMC, SENS, TARGETS = CFG["prior"], CFG["mcmc"], CFG["sensitivity"], CFG["diagnostics"]
TABLES, RESULTS = CFG["paths"]["tables"], CFG["paths"]["results"]
NUMBERS = TABLES.parent / "numbers"
LEVELS = [0.99, 0.999]  # levels compared across variants
rng_prior, rng_fits = np.random.default_rng(MCMC["seed"]).spawn(2)


# %% 1. Prior predictive check: draw theta from the prior, look at the daily sd and the 99% VaR it implies
theta = sample_prior(SENS["n_prior_draws"], PRIOR, rng_prior)
implied = {r"$\nu$": theta[:, 2],
           "Daily sd (%)": 100 * theta[:, 1] * np.sqrt(theta[:, 2] / (theta[:, 2] - 2)),
           "VaR 99% (%)": 100 * var_t(theta, 0.99)}
rows = [[name, *(f"{np.quantile(x, q):.3g}" for q in (0.025, 0.5, 0.975))] for name, x in implied.items()]
write_table(["Implied by the prior", "2.5%", "Median", "97.5%"], rows, TABLES / "prior_predictive.tex")
print("Prior predictive check (2.5%, median, 97.5%)")
print("\n".join(f"  {row[0]:16s} " + "  ".join(f"{c:>8s}" for c in row[1:]) for row in rows))
numbers = {"prior-var99-lo": rows[2][1], "prior-var99-med": rows[2][2], "prior-var99-hi": rows[2][3],
           "prior-sd-lo": rows[1][1], "prior-sd-hi": rows[1][3], "prior-n-draws": f"{SENS['n_prior_draws']:,}"}

# %% 2. Sensitivity variants: other priors, a looser bound on nu, and a shorter window
r_all = load_returns(CFG)
variants = {"Baseline": (PRIOR, r_all),
            r"$\nu$ prior rate 0.05": (PRIOR | {"nu_rate": SENS["nu_rate_low"]}, r_all),
            r"$\nu$ prior rate 0.5": (PRIOR | {"nu_rate": SENS["nu_rate_high"]}, r_all),
            r"$\mu$ prior sd 0.01 (pitch)": (PRIOR | {"mu_sd": SENS["mu_sd_pitch"]}, r_all),
            r"$\sigma$ prior scale 0.05": (PRIOR | {"sigma_scale": SENS["sigma_scale_wide"]}, r_all),
            r"$\nu > 1$ instead of $\nu > 2$": (PRIOR | {"nu_min": SENS["nu_min_low"]}, r_all),
            f"Window {SENS['window_start'][:4]}--{r_all.index[-1].year}": (PRIOR, r_all[SENS["window_start"]:])}

# %% 3. Refit the Student-t model under each variant (the baseline reuses the draws of 05_fit_real_data.py)
rows, risks, keys = [], {}, ["base", "nu-low", "nu-high", "mu-pitch", "sigma-wide", "numin1", "window"]
for (name, (prior, r)), key, rng in zip(variants.items(), keys, rng_fits.spawn(len(variants))):
    if name == "Baseline":
        draws = np.load(RESULTS / "posterior_draws.npz")["student_t"]
        rhat = "--"
    else:
        r = r.to_numpy()
        chains, accept = run_chains(lambda phi: log_posterior(phi, r, prior), [np.median(r), np.log(r.std()), 0.0], MCMC, rng)
        draws = to_natural(chains[:, MCMC["burn_in"]:], prior["nu_min"])
        summaries = [summarize(draws[..., i]) for i in range(3)]
        rhat = f"{max(s['rhat'] for s in summaries):.4f}"
        print(f"  {name}: all targets {'PASS' if all(check_targets(summaries, accept, TARGETS).values()) else 'FAIL'}")
    nu = summarize(draws[..., 2])
    risk = [100 * v for a in LEVELS for v in predictive_t(draws, a)]  # VaR 99, ES 99, VaR 99.9, ES 99.9 (%)
    risks[key] = np.array([nu["mean"], *risk])
    rows.append([name, f"{nu['mean']:.2f} [{nu['lo']:.2f}, {nu['hi']:.2f}]", *(f"{v:.2f}" for v in risk), rhat])
    numbers |= {f"sens-{key}-nu": f"{nu['mean']:.2f}", f"sens-{key}-var99": f"{risk[0]:.2f}",
                f"sens-{key}-es99": f"{risk[1]:.2f}", f"sens-{key}-es99.9": f"{risk[3]:.2f}"}
write_table(["Variant", r"$\nu$: mean [95% interval]", "VaR 99%", "ES 99%", "VaR 99.9%", "ES 99.9%", r"max $\hat R$"],
            rows, TABLES / "sensitivity.tex")
# largest relative change of nu or of a risk measure across the prior variants (all variants but the window)
prior_dev = max(np.abs(risks[k] / risks["base"] - 1).max() for k in keys[1:-1])
numbers["sens-prior-maxdev"] = f"{100 * prior_dev:.1f}"
print(f"Largest change across the prior variants: {100 * prior_dev:.1f}%")
write_numbers(numbers, NUMBERS / "sensitivity.tex")
print("\nSensitivity (nu, predictive VaR 99%, ES 99%, VaR 99.9%, ES 99.9% in %, max R-hat)")
print("\n".join(f"  {row[0].replace('$', '').replace(chr(92), ''):32s} " + "  ".join(f"{c:>20s}" for c in row[1:]) for row in rows))
print("\n->", TABLES / "prior_predictive.tex,", TABLES / "sensitivity.tex,", NUMBERS / "sensitivity.tex")
