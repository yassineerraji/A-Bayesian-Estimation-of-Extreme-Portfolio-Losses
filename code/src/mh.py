"""Random-walk Metropolis–Hastings, and the full sampling recipe: MAP, pilot run, frozen proposal, several chains."""

import numpy as np
from scipy.optimize import minimize


# ======================================================================
# [COURSE] Metropolis–Hastings: Ch. 6, Sec. 6.2, Algorithm 4 (random walk: Sec. 6.3)
# Accept with probability min{1, π̃(θ')/π̃(θ)}: only ratios are needed.
# The proposal is symmetric, so the q-ratio cancels. Rejections are recorded.
# ======================================================================
def rw_metropolis(log_target, phi0, chol, n_iter, rng):
    """Random-walk Metropolis with Gaussian proposal phi' = phi + chol @ z, z ~ N(0, I).

    Returns the draws, shape (n_iter, d), and the acceptance rate.
    """
    phi, lp = np.asarray(phi0, dtype=float), log_target(phi0)
    draws, accepted = np.empty((n_iter, len(phi))), 0
    for t in range(n_iter):
        proposal = phi + chol @ rng.standard_normal(len(phi))
        lp_proposal = log_target(proposal)
        if np.log(rng.uniform()) < lp_proposal - lp:  # accept, on the log scale
            phi, lp, accepted = proposal, lp_proposal, accepted + 1
        draws[t] = phi  # after a rejection, the current value is recorded again
    return draws, accepted / n_iter


# ======================================================================
# [COURSE] Tuning: Ch. 6, Sec. 6.5 (MAP: Ch. 4, Def. 4.1)
# Start from the MAP with the Laplace scale, run pilots, then freeze the
# proposal covariance (2.38^2 / d) Σ̂; aim for an acceptance rate of 0.2–0.5.
# ======================================================================
def run_chains(log_target, phi_init, starts, mcmc, rng):
    """Tune the random-walk proposal, then run one chain from each starting point.

    log_target: unnormalized log density of phi; phi_init: where the MAP search starts;
    starts: (chains, d) dispersed starting points; mcmc: the [mcmc] settings; rng: NumPy generator.
    Returns the chains, shape (chains, n_iter, d) with burn-in included, and their acceptance rates.
    """
    d = len(phi_init)
    # 1. MAP, with the inverse Hessian at the mode (Laplace) as a first proposal covariance
    fit = minimize(lambda phi: -log_target(phi), phi_init, method="BFGS")
    chol = np.linalg.cholesky(2.38**2 / d * fit.hess_inv)
    # 2. Pilot runs from the MAP: each re-estimates the proposal covariance from its own draws; the last one is frozen
    # DECISION: several rounds, because BFGS's inverse Hessian can be a poor first proposal (it stops with
    # "precision loss"): with one round, 3 of 10 simulated data sets gave an ESS below 30; with 3 rounds, all above 5,000.
    phi = fit.x
    for _ in range(mcmc["pilot_rounds"]):
        pilot, _ = rw_metropolis(log_target, phi, chol, mcmc["pilot_iter"], rng)
        chol, phi = np.linalg.cholesky(2.38**2 / d * np.cov(pilot.T)), pilot[-1]
    # 3. Main run: one chain per starting point, each with its own independent random stream
    runs = [rw_metropolis(log_target, s, chol, mcmc["n_iter"], g) for s, g in zip(starts, rng.spawn(len(starts)))]
    return np.array([draws for draws, _ in runs]), np.array([rate for _, rate in runs])
