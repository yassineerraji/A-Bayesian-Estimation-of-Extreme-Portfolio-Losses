"""Risk measures of the one-day loss L = -r: Value-at-Risk and Expected Shortfall.

With r = mu + scale * Z, the loss is L = -mu + scale * Z' where Z' = -Z has the same symmetric standard
distribution (Student-t with nu degrees of freedom, or Gaussian).
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq


# ======================================================================
# [COURSE] Posterior of a function of the parameters: Ch. 1, Def. 1.4, eq. (1.7)
# VaR and ES are functions h(theta); applied to every posterior draw, they
# give draws from the posterior of VaR and ES.
# ======================================================================
def var_t(theta, alpha):
    """Student-t VaR at level alpha (the loss exceeded with probability 1 - alpha).

    theta: array whose last axis is (mu, sigma, nu): one parameter vector or many draws.
    """
    mu, sigma, nu = theta[..., 0], theta[..., 1], theta[..., 2]
    return -mu + sigma * stats.t.ppf(alpha, nu)


def es_t(theta, alpha):
    """Student-t ES at level alpha (the mean loss beyond the VaR; finite for nu > 1). theta as in var_t."""
    mu, sigma, nu = theta[..., 0], theta[..., 1], theta[..., 2]
    q = stats.t.ppf(alpha, nu)
    return -mu + sigma * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)


def var_gauss(theta, alpha):
    """Gaussian VaR at level alpha; theta: array whose last axis is (mu, s)."""
    return -theta[..., 0] + theta[..., 1] * stats.norm.ppf(alpha)


def es_gauss(theta, alpha):
    """Gaussian ES at level alpha; theta as in var_gauss."""
    return -theta[..., 0] + theta[..., 1] * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)


def tail_t(c, nu):
    """P(Z > c) and E[Z 1{Z > c}] for a standard Student-t Z with nu degrees of freedom."""
    return stats.t.sf(c, nu), stats.t.pdf(c, nu) * (nu + c**2) / (nu - 1)


def tail_gauss(c):
    """P(Z > c) and E[Z 1{Z > c}] for a standard Gaussian Z."""
    return stats.norm.sf(c), stats.norm.pdf(c)


# ======================================================================
# [COURSE] Posterior predictive + Rao–Blackwell: Ch. 3, Example 3.2; Prop. 3.4
# The predictive loss is a mixture over the posterior draws. Instead of
# simulating losses, we average their exact conditional tail probabilities
# and tail expectations given each draw: same target, smaller variance.
# ======================================================================
def predictive_var_es(mu, scale, tail, alpha):
    """VaR and ES of the posterior predictive loss, a mixture over the draws (mu, scale).

    tail(c) returns P(Z > c) and E[Z 1{Z > c}] for each draw's standard distribution.
    VaR solves mean_i P_i(L > v) = 1 - alpha; ES = mean_i E_i[L 1{L > v}] / (1 - alpha).
    """
    exceed = lambda v: tail((v + mu) / scale)[0].mean() - (1 - alpha)  # noqa: E731
    var = brentq(exceed, -1, 1)  # a one-day log loss lies well inside (-100%, 100%)
    p, m = tail((var + mu) / scale)
    return var, np.mean(-mu * p + scale * m) / (1 - alpha)


def predictive_t(theta, alpha):
    """Predictive VaR and ES of the Student-t model; theta: posterior draws, last axis (mu, sigma, nu)."""
    return predictive_var_es(theta[..., 0], theta[..., 1], lambda c: tail_t(c, theta[..., 2]), alpha)


def predictive_gauss(theta, alpha):
    """Predictive VaR and ES of the Gaussian model; theta: posterior draws, last axis (mu, s)."""
    return predictive_var_es(theta[..., 0], theta[..., 1], tail_gauss, alpha)
