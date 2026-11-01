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
# [COURSE] Laplace approximation: Ch. 3, Example 3.4; Ch. 6, Sec. 6.5
# Near a mode, the posterior is close to a Gaussian whose covariance is the
# inverse of minus the Hessian (curvature) of log π̃ at the mode.
# ======================================================================
def laplace_cov(log_target, mode, h=1e-4):
    """Inverse of minus the Hessian of log_target at the mode, by central finite differences."""
    E = h * np.eye(len(mode))
    H = [[log_target(mode + a + b) - log_target(mode + a - b) - log_target(mode - a + b) + log_target(mode - a - b)
          for b in E] for a in E]
    return np.linalg.inv(-np.array(H) / (4 * h**2))


# ======================================================================
# [COURSE] Tuning: Ch. 6, Sec. 6.5 (MAP: Ch. 4, Def. 4.1)
# Start from the MAP with the Laplace scale, run pilots, then freeze the
# proposal covariance (2.38^2 / d) Σ̂; aim for an acceptance rate of 0.2–0.5.
# Several chains from dispersed starting points: Ch. 6, Sec. 6.6.
# ======================================================================
def run_chains(log_target, phi_init, mcmc, rng):
    """Tune the random-walk proposal, then run several chains from dispersed starting points.

    log_target: unnormalized log density of phi; phi_init: where the MAP search starts;
    mcmc: the [mcmc] settings; rng: NumPy generator.
    Returns the chains, shape (chains, n_iter, d) with burn-in included, and their acceptance rates.
    """
    d, step = len(phi_init), 2.38 / np.sqrt(len(phi_init))  # proposal covariance = step^2 * posterior covariance
    # 1. MAP, with the Laplace covariance at the mode as a first estimate of the posterior covariance
    # DECISION: finite-difference Hessian, not BFGS's own inverse-Hessian estimate, which can be far off (up to 70
    # times too small for log(nu - nu_min) on simulated data) or untouched when the search starts at the mode.
    fit = minimize(lambda phi: -log_target(phi), phi_init, method="BFGS")
    L = np.linalg.cholesky(laplace_cov(log_target, fit.x))
    # 2. Pilot runs from the MAP: each re-estimates the posterior covariance from its own draws; the last one is frozen
    # DECISION: several rounds, so that the frozen covariance never depends on the Laplace approximation alone.
    phi = fit.x
    for _ in range(mcmc["pilot_rounds"]):
        pilot, _ = rw_metropolis(log_target, phi, step * L, mcmc["pilot_iter"], rng)
        L, phi = np.linalg.cholesky(np.cov(pilot.T)), pilot[-1]
    # 3. Dispersed starting points: the MAP plus start_spread posterior standard deviations in random directions
    # DECISION: overdispersed around the MAP, not drawn from the prior: a prior draw with nu close to nu_min starts on
    # an almost flat ridge (nu barely changes the likelihood there) and can need thousands of iterations to leave it.
    starts = fit.x + mcmc["start_spread"] * rng.standard_normal((mcmc["chains"], d)) @ L.T
    # 4. Main run: one chain per starting point, each with its own independent random stream
    runs = [rw_metropolis(log_target, s, step * L, mcmc["n_iter"], g) for s, g in zip(starts, rng.spawn(len(starts)))]
    return np.array([draws for draws, _ in runs]), np.array([rate for _, rate in runs])
