"""Student-t model for daily log-returns: likelihood, prior and posterior on the sampling scale.

The sampler works on phi = (mu, eta, lam), with sigma = exp(eta) and nu = nu_min + exp(lam),
so that every real vector phi is a valid parameter.
"""

import numpy as np
from scipy.special import gammaln


def to_natural(phi, nu_min):
    """Map phi = (mu, log sigma, log(nu - nu_min)) to theta = (mu, sigma, nu); works on any array whose last axis is phi."""
    phi = np.asarray(phi)
    return np.stack([phi[..., 0], np.exp(phi[..., 1]), nu_min + np.exp(phi[..., 2])], axis=-1)


def to_phi(theta, nu_min):
    """Map theta = (mu, sigma, nu) to the sampling scale phi; inverse of to_natural."""
    theta = np.asarray(theta)
    return np.stack([theta[..., 0], np.log(theta[..., 1]), np.log(theta[..., 2] - nu_min)], axis=-1)


def sample_prior(n, prior, rng):
    """Draw n parameter vectors theta = (mu, sigma, nu) directly from the prior; returns shape (n, 3)."""
    return np.column_stack([rng.normal(0, prior["mu_sd"], n),
                            np.abs(rng.normal(0, prior["sigma_scale"], n)),
                            prior["nu_min"] + rng.exponential(1 / prior["nu_rate"], n)])


# ======================================================================
# [COURSE] Log scale against underflow: Ch. 1, Remark 1.3
# A product of 5,031 densities rounds to 0, so we always sum log densities.
# ======================================================================
def log_likelihood(mu, sigma, nu, r):
    """Student-t log-likelihood of the returns r, summed over days (normalizing constants included)."""
    z = (r - mu) / sigma
    return (len(r) * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma))
            - (nu + 1) / 2 * np.log1p(z**2 / nu).sum())


# ======================================================================
# [COURSE] Change of variables (Jacobian): Ch. 2, Prop. 2.1; Ch. 6, Sec. 6.2
# Sampling log sigma and log(nu - nu_min) multiplies the prior density by
# |d sigma / d eta| = sigma and |d nu / d lam| = nu - nu_min, i.e. adds eta + lam.
# ======================================================================
def log_prior(phi, prior):
    """Log prior density of phi up to a constant, Jacobian included."""
    mu, sigma, nu = to_natural(phi, prior["nu_min"])
    return (-0.5 * (mu / prior["mu_sd"]) ** 2              # mu ~ Normal(0, mu_sd^2)
            - 0.5 * (sigma / prior["sigma_scale"]) ** 2    # sigma ~ HalfNormal(sigma_scale)
            - prior["nu_rate"] * (nu - prior["nu_min"])    # nu - nu_min ~ Exponential(nu_rate)
            + phi[1] + phi[2])                             # Jacobian


# ======================================================================
# [COURSE] Unnormalized posterior: Ch. 1, Sec. 1.2.2, Def. 1.3, eq. (1.6)
# pi(theta | y) ∝ p(y | theta) pi(theta); the evidence p(y) is never needed.
# ======================================================================
def log_posterior(phi, r, prior):
    """Unnormalized log posterior of phi given the returns r."""
    mu, sigma, nu = to_natural(phi, prior["nu_min"])
    return log_likelihood(mu, sigma, nu, r) + log_prior(phi, prior)
