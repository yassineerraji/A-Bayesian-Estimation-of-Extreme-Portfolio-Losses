"""Cross-check of the Metropolis–Hastings results with importance sampling, a second and independent course method.

Self-normalized importance sampling of the Student-t posterior, with a multivariate Student-t proposal centred at
the MAP with the Laplace covariance; its tails are heavier than the posterior's, so the weights stay bounded.

Reads:     config.toml; data/returns.csv; results/posterior_draws.npz (MH draws)
Produces:  report/tables/crosscheck.tex     posterior means of the parameters and of VaR/ES: MH against IS
           report/numbers/crosscheck.tex    numbers quoted in the text
Run:       python scripts/09_crosscheck_is.py   (about 10 s)
"""

# Imports
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.diagnostics import summarize  # noqa: E402
from src.mh import laplace_cov  # noqa: E402
from src.model import log_posterior, to_natural  # noqa: E402
from src.report import write_numbers, write_table  # noqa: E402
from src.risk import es_t, var_t  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
PRIOR, CC = CFG["prior"], CFG["crosscheck"]
TABLES, RESULTS = CFG["paths"]["tables"], CFG["paths"]["results"]
NUMBERS = TABLES.parent / "numbers"
rng = np.random.default_rng(CFG["mcmc"]["seed"])
QUANTITIES = {r"$\mu$ (%)": lambda th: 100 * th[..., 0], r"$\sigma$ (%)": lambda th: 100 * th[..., 1],
              r"$\nu$": lambda th: th[..., 2],
              "VaR 99% (%)": lambda th: 100 * var_t(th, 0.99), "ES 99% (%)": lambda th: 100 * es_t(th, 0.99),
              "VaR 99.9% (%)": lambda th: 100 * var_t(th, 0.999), "ES 99.9% (%)": lambda th: 100 * es_t(th, 0.999)}


# %% 1. Proposal: multivariate Student-t centred at the MAP, with the Laplace covariance
# ======================================================================
# [COURSE] Importance sampling with a Laplace Student-t proposal: Ch. 3, Example 3.4
# Centre at the mode, scale from the curvature there, Student-t tails so that
# the proposal has heavier tails than the target and the weights stay bounded.
# ======================================================================
r = load_returns(CFG).to_numpy()
log_target = lambda phi: log_posterior(phi, r, PRIOR)  # noqa: E731
mode = minimize(lambda phi: -log_target(phi), [np.median(r), np.log(r.std()), 0.0], method="BFGS").x
proposal = stats.multivariate_t(loc=mode, shape=laplace_cov(log_target, mode), df=CC["df"], seed=rng)

# %% 2. Draw from the proposal and compute the self-normalized weights, on the log scale
# ======================================================================
# [COURSE] Self-normalized importance sampling and its ESS: Ch. 3, Def. 3.3
# w_i ∝ π̃(φ_i)/g(φ_i), normalized with log-sum-exp (Ch. 1, Remark 1.3);
# ESS = 1 / sum of squared normalized weights.
# ======================================================================
phi = proposal.rvs(CC["n_draws"])
log_w = np.array([log_target(p) for p in phi]) - proposal.logpdf(phi)
w = np.exp(log_w - logsumexp(log_w))
ess = 1 / np.sum(w**2)
print(f"Importance sampling: {CC['n_draws']:,} draws, ESS of the weights {ess:,.0f} ({100 * ess / CC['n_draws']:.0f}%)")

# %% 3. Compare posterior means: MH against IS, each with its Monte Carlo error
# ======================================================================
# [COURSE] Two different algorithms must agree: Ch. 6, Example 6.2
# ======================================================================
theta_is, theta_mh = to_natural(phi, PRIOR["nu_min"]), np.load(RESULTS / "posterior_draws.npz")["student_t"]
rows, gaps = [], []
for name, f in QUANTITIES.items():
    mh, x = summarize(f(theta_mh)), f(theta_is)
    mean_is = np.sum(w * x)
    se_is = np.sqrt(np.sum(w * (x - mean_is) ** 2) / ess)  # posterior sd / sqrt(ESS)
    gaps.append(abs(mh["mean"] - mean_is) / np.hypot(mh["mcse"], se_is))
    rows.append([name, f"{mh['mean']:.4g}", f"{mh['mcse']:.1g}", f"{mean_is:.4g}", f"{se_is:.1g}", f"{gaps[-1]:.1f}"])
write_table(["Quantity", "MH mean", "MCSE", "IS mean", "IS error", "Gap (in errors)"], rows, TABLES / "crosscheck.tex")
print("Posterior means (MH mean, MCSE | IS mean, error | gap in combined errors)")
print("\n".join(f"  {row[0].replace('$', '').replace(chr(92), ''):14s} " + "  ".join(f"{c:>10s}" for c in row[1:]) for row in rows))
write_numbers({"is-ess": f"{ess:,.0f}", "is-ess-pct": f"{100 * ess / CC['n_draws']:.0f}",
               "is-n": f"{CC['n_draws']:,}", "is-max-gap": f"{max(gaps):.1f}"}, NUMBERS / "crosscheck.tex")
print("\n->", TABLES / "crosscheck.tex,", NUMBERS / "crosscheck.tex")
