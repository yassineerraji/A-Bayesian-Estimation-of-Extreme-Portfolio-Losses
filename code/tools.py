"""Functions shared by our notebooks.

Every notebook imports what it needs from this file (`from tools import ...`), so that each
function is written once. The file grows with the project; its sections are:

    Figures and tables
        COLOR_T, COLOR_GAUSS, COLOR_OTHER, COLOR_DATA   our colours, one per role
        set_style()                                     the figure style of every notebook
        save_figure(fig, name)                          save a figure in ../report/figures/
        save_table(table, name)                         save a table in ../report/tables/

    Model (Student-t returns, notebook 02)
        PRIOR                                           the prior settings
        to_theta(phi), to_phi(theta)                    theta = (mu, sigma, nu) <-> sampling scale phi
        log_likelihood(theta, y)                        Student-t log-likelihood
        log_prior(phi), log_posterior(phi, y)           log prior (Jacobian included), log posterior
        log_posterior_gauss(phi, y)                     log posterior of the Gaussian benchmark

    Sampler (notebooks 02-04)
        metropolis(log_target, phi0, L, n_iter, rng)    random-walk Metropolis-Hastings
        laplace_cov(log_target, mode)                   Laplace approximation of the covariance
        find_map(log_target, phi_start)                 the MAP and the Laplace covariance there
        run_chains(log_target, phi_start, rng)          our full recipe: MAP, pilot runs, 4 chains
        BURN_IN                                         iterations discarded at the start of a chain
        independence_sampler(log_target, mode, cov, ...) MH with a fixed Student-t proposal

    Diagnostics (notebooks 02-03)
        autocorr(x, k), split_rhat(chains), ess(chains) autocorrelation, R-hat, effective sample size
        summarize(chains)                               posterior summary of one quantity
        convergence_checks(table, accept_rates)         PASS/FAIL of our convergence targets
        plot_diagnostics(draws, labels, truth)          traces and autocorrelations of every chain

    Risk measures (notebooks 02-05)
        var_t(theta, alpha), es_t(theta, alpha)         Student-t VaR and ES of the one-day loss
        var_gauss(theta, alpha), es_gauss(theta, alpha) Gaussian VaR and ES
        tail_t(theta, v), tail_gauss(theta, v)          exact probability and mean beyond a loss v
        predictive_var_es(theta_draws, alpha, model)    VaR and ES of the posterior predictive loss
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import gammaln


# ----------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------

# Our colours: the first three colours of the Okabe-Ito palette, which stay distinct for
# colour-blind readers (checked with a colour-vision simulator), plus a neutral grey for data.
# Exact or true reference values are drawn in black.
COLOR_T = "#0072B2"      # Student-t model (blue)
COLOR_GAUSS = "#D55E00"  # Gaussian model (vermillion)
COLOR_OTHER = "#009E73"  # a third series when we need one (green)
COLOR_DATA = "#6E6E6E"   # observed data (grey)

# Where the report picks up our figures and tables (paths relative to the code/ folder)
FIGURES = Path("../report/figures")
TABLES = Path("../report/tables")


def set_style():
    """Apply our figure style: report-sized figures, small fonts, a light grid, no top/right frame."""
    plt.rcParams.update({
        "figure.figsize": (6.3, 2.6),  # 6.3 inches = 16 cm, the text width of the report
        "font.size": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,  # spaces the panels automatically
        "savefig.dpi": 300,  # resolution of the few rasterized layers (long traces); the rest is vector
    })


def save_figure(fig, name):
    """Save a figure as a PDF in ../report/figures/ (the notebook still shows it)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    # no creation date in the file, so that rerunning a notebook gives an identical PDF
    fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"CreationDate": None})


def latex_cell(value, fmt=None):
    """Text of one table cell for LaTeX: escape % and &, and give numbers a true minus sign.

    fmt: optional format for decimal numbers, e.g. "{:.3f}" to always show 3 decimals.
    """
    if fmt is not None and isinstance(value, (float, np.floating)):
        text = fmt.format(value)
    else:
        text = str(value)
    text = text.replace("%", r"\%").replace("&", r"\&")
    if text.startswith("-") and text[1:2].isdigit():  # a negative number, as a number or as text
        text = "$-$" + text[1:]
    return text


def save_table(table, name, fmt=None):
    """Save a pandas DataFrame as a LaTeX table (booktabs style) in ../report/tables/.

    The index becomes the first column. Decimal numbers are written with fmt (e.g. "{:.3f}")
    when it is given, and as they are displayed otherwise.
    """
    header = [table.index.name or ""] + list(table.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(table.columns) + "}",
             r"\toprule",
             " & ".join(latex_cell(h) for h in header) + r" \\",
             r"\midrule"]
    for label, row in table.iterrows():
        cells = [latex_cell(label)] + [latex_cell(value, fmt) for value in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Model: y_t iid Student-t with location mu, scale sigma and nu degrees of freedom
# ----------------------------------------------------------------------

# Prior settings, for daily log-returns in decimals (0.01 = 1%):
#   mu ~ Normal(0, mu_sd^2),  sigma ~ HalfNormal(sigma_scale),  nu - nu_min ~ Exponential(rate nu_rate)
# A dict, so that notebook 06 can pass other priors to the same functions.
PRIOR = {"mu_sd": 0.001, "sigma_scale": 0.02, "nu_min": 2.0, "nu_rate": 0.1}


def to_theta(phi, prior=PRIOR):
    """Map phi = (mu, log sigma, log(nu - nu_min)) to theta = (mu, sigma, nu).

    phi can be one vector or an array of draws whose last axis holds the three coordinates.
    """
    phi = np.asarray(phi)
    mu = phi[..., 0]
    sigma = np.exp(phi[..., 1])
    nu = prior["nu_min"] + np.exp(phi[..., 2])
    return np.stack([mu, sigma, nu], axis=-1)


def to_phi(theta, prior=PRIOR):
    """Map theta = (mu, sigma, nu) to phi = (mu, log sigma, log(nu - nu_min)); the inverse of to_theta."""
    theta = np.asarray(theta)
    mu = theta[..., 0]
    log_sigma = np.log(theta[..., 1])
    log_nu = np.log(theta[..., 2] - prior["nu_min"])
    return np.stack([mu, log_sigma, log_nu], axis=-1)


# ======================================================================
# [COURSE] Log scale against underflow: Ch. 1, Sec. 1.3.1, Remark 1.3 (p. 20)
# A product of 5,031 densities rounds to 0 in floating point, so we always sum log densities.
# ======================================================================
def log_likelihood(theta, y):
    """Student-t log-likelihood l(theta) = sum over t of log p(y_t | mu, sigma, nu), constants included."""
    mu, sigma, nu = theta
    z = (y - mu) / sigma
    log_density = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma)
                   - (nu + 1) / 2 * np.log1p(z**2 / nu))
    return log_density.sum()


# ======================================================================
# [COURSE] Change of variables: Ch. 2, Sec. 2.4.2, Proposition 2.1 (p. 34);
# constrained parameters in MH: Ch. 6, Sec. 6.2 (p. 69)
# We sample phi = (mu, log sigma, log(nu - nu_min)), which can take any real value. The density
# of phi is the density of theta times |d sigma / d log sigma| = sigma and |d nu / d log(nu - nu_min)|
# = nu - nu_min: on the log scale, we add log sigma + log(nu - nu_min), the Jacobian.
# ======================================================================
def log_prior(phi, prior=PRIOR):
    """Log prior density of phi = (mu, log sigma, log(nu - nu_min)), up to a constant, Jacobian included."""
    mu, sigma, nu = to_theta(phi, prior)
    log_p_mu = -0.5 * (mu / prior["mu_sd"]) ** 2              # mu ~ Normal(0, mu_sd^2)
    log_p_sigma = -0.5 * (sigma / prior["sigma_scale"]) ** 2  # sigma ~ HalfNormal(sigma_scale)
    log_p_nu = -prior["nu_rate"] * (nu - prior["nu_min"])     # nu - nu_min ~ Exponential(rate)
    log_jacobian = phi[1] + phi[2]                            # log sigma + log(nu - nu_min)
    return log_p_mu + log_p_sigma + log_p_nu + log_jacobian


# ======================================================================
# [COURSE] Unnormalized posterior: Ch. 1, Sec. 1.2.2, Definition 1.3 (p. 15)
# pi(theta | y) is proportional to p(y | theta) pi(theta). Metropolis-Hastings only uses ratios
# of the posterior, so the evidence p(y) is never needed.
# ======================================================================
def log_posterior(phi, y, prior=PRIOR):
    """Unnormalized log posterior of phi given the returns y, for the Student-t model."""
    return log_likelihood(to_theta(phi, prior), y) + log_prior(phi, prior)


# The Gaussian benchmark: y_t iid Normal(mu, s^2), with the same priors as the Student-t's mu and sigma
# (mu ~ Normal(0, mu_sd^2), s ~ HalfNormal(sigma_scale)), so that the two models differ only in
# their tails. It is sampled on phi = (mu, log s), with the Jacobian |ds / d log s| = s.
def log_posterior_gauss(phi, y, prior=PRIOR):
    """Unnormalized log posterior of the Gaussian model, phi = (mu, log s), Jacobian included."""
    mu = phi[0]
    s = np.exp(phi[1])
    log_lik = -len(y) * np.log(s) - np.sum((y - mu) ** 2) / (2 * s**2)  # up to a constant
    log_p_mu = -0.5 * (mu / prior["mu_sd"]) ** 2
    log_p_s = -0.5 * (s / prior["sigma_scale"]) ** 2
    log_jacobian = phi[1]  # log s
    return log_lik + log_p_mu + log_p_s + log_jacobian


# ----------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------

# ======================================================================
# [COURSE] Metropolis-Hastings: Ch. 6, Sec. 6.2, Algorithm 4 (p. 69); random walk: Sec. 6.3 (p. 69)
# Propose phi' = phi + L z and accept it with probability min{1, pi~(phi') / pi~(phi)}: the
# proposal is symmetric, so the q-ratio of Algorithm 4 cancels. We compare on the log scale.
# A rejected proposal records the current value again: repeats are part of the output.
# ======================================================================
def metropolis(log_target, phi0, L, n_iter, rng):
    """Random-walk Metropolis-Hastings with Gaussian proposal phi' = phi + L z, z ~ N(0, I).

    log_target: function returning the unnormalized log density of phi
    phi0: starting point; L: matrix such that L L^T is the proposal covariance
    Returns the draws, shape (n_iter, d), and the acceptance rate.
    """
    phi = np.array(phi0, dtype=float)
    log_p = log_target(phi)
    d = len(phi)
    draws = np.empty((n_iter, d))
    n_accepted = 0
    for t in range(n_iter):
        proposal = phi + L @ rng.standard_normal(d)
        log_p_proposal = log_target(proposal)
        if np.log(rng.uniform()) < log_p_proposal - log_p:  # accept
            phi = proposal
            log_p = log_p_proposal
            n_accepted += 1
        draws[t] = phi  # after a rejection, the current value is recorded again
    return draws, n_accepted / n_iter


# ======================================================================
# [COURSE] Laplace approximation: Ch. 6, Sec. 6.5 (p. 72); Ch. 3, Example 3.4 (p. 45)
# Near its mode, the posterior is close to a Gaussian whose covariance is the inverse of
# minus the matrix of second derivatives (the Hessian) of log pi~ at the mode.
# ======================================================================
def laplace_cov(log_target, mode, h=1e-4):
    """Inverse of minus the Hessian of log_target at the mode, by central finite differences."""
    d = len(mode)
    steps = h * np.eye(d)
    H = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            H[i, j] = (log_target(mode + steps[i] + steps[j]) - log_target(mode + steps[i] - steps[j])
                       - log_target(mode - steps[i] + steps[j])
                       + log_target(mode - steps[i] - steps[j])) / (4 * h**2)
    return np.linalg.inv(-H)


# ======================================================================
# [COURSE] The MAP: Ch. 4, Sec. 4.1, Definition 4.1 (p. 51); a good starting point for every
# sampler (App. C.3, p. 90)
# ======================================================================
def find_map(log_target, phi_start):
    """The posterior mode (MAP) on the sampling scale, and the Laplace covariance there.

    Returns (mode, covariance). The search starts at phi_start.
    """
    def minus_log_target(phi):
        return -log_target(phi)
    mode = minimize(minus_log_target, phi_start, method="BFGS").x
    # DECISION: a finite-difference Hessian, not the BFGS estimate, which can be far off (found in v1)
    return mode, laplace_cov(log_target, mode)


BURN_IN = 2_000  # iterations discarded at the start of every chain (10% of a 20,000-iteration chain)


# ======================================================================
# [COURSE] Tuning: Ch. 6, Sec. 6.5 (p. 72); the MAP: Ch. 4, Definition 4.1 (p. 51);
# several chains from dispersed starting points: Sec. 6.6 (p. 73)
# Start from the MAP with the Laplace scale, run pilots, then freeze the proposal covariance
# at (2.38^2 / d) Sigma_hat, which aims for an acceptance rate between 0.2 and 0.5.
# ======================================================================
def run_chains(log_target, phi_start, rng, n_chains=4, n_iter=20_000, n_pilot=5_000,
               pilot_rounds=3, spread=5):
    """Our full sampling recipe: find the MAP, tune the proposal with pilot runs, run several chains.

    log_target: unnormalized log density of phi; phi_start: where the search for the MAP starts
    n_chains chains of n_iter iterations; pilot_rounds pilot runs of n_pilot iterations;
    spread: how many posterior standard deviations the starting points lie from the MAP.
    Returns the chains, shape (n_chains, n_iter, d), burn-in included, and their acceptance rates.
    """
    d = len(phi_start)
    scale = 2.38 / np.sqrt(d)  # proposal covariance = scale^2 x posterior covariance

    # 1. The MAP (on the sampling scale), and the Laplace covariance as a first guess
    mode, cov = find_map(log_target, phi_start)

    # 2. Pilot runs from the MAP: each one re-estimates the posterior covariance from its own draws
    # DECISION: 3 rounds, so that the frozen proposal never rests on the Laplace guess alone (v1)
    phi = mode
    for _ in range(pilot_rounds):
        pilot, _ = metropolis(log_target, phi, scale * np.linalg.cholesky(cov), n_pilot, rng)
        cov = np.atleast_2d(np.cov(pilot, rowvar=False))  # atleast_2d: also works when d = 1
        phi = pilot[-1]

    # 3. Freeze the proposal: covariance (2.38^2 / d) Sigma_hat
    L = scale * np.linalg.cholesky(cov)

    # 4. One chain per starting point, each `spread` posterior sd away from the MAP in a random direction
    # DECISION: starts spread around the MAP, not drawn from the prior: a prior draw with nu close
    # to 2 starts on an almost flat ridge and can need thousands of iterations to leave it (v1)
    chains = np.empty((n_chains, n_iter, d))
    accept_rates = np.empty(n_chains)
    for c in range(n_chains):
        start = mode + spread * np.linalg.cholesky(cov) @ rng.standard_normal(d)
        chains[c], accept_rates[c] = metropolis(log_target, start, L, n_iter, rng)
    return chains, accept_rates


# ======================================================================
# [COURSE] Independence sampler: Ch. 6, Sec. 6.3 (p. 70); its proposal, the Laplace approximation
# with Student-t tails: Ch. 3, Example 3.4 (p. 45)
# Proposals come from a fixed density g, so the acceptance probability min{1, pi~(phi') g(phi) /
# (pi~(phi) g(phi'))} is the ratio of the importance weights w = pi~ / g of the two points.
# ======================================================================
def independence_sampler(log_target, mode, cov, n_iter, rng, df=4):
    """Independence Metropolis-Hastings with a fixed Student-t proposal g centred at the MAP.

    mode, cov: centre and scale matrix of g (the MAP and the Laplace covariance); df: its tails.
    Returns the draws, shape (n_iter, d), and the acceptance rate.
    """
    g = stats.multivariate_t(loc=mode, shape=cov, df=df)
    proposals = g.rvs(size=n_iter, random_state=rng)
    log_g = g.logpdf(proposals)  # computed once for all proposals
    phi = mode
    log_w = log_target(mode) - g.logpdf(mode)
    draws = np.empty((n_iter, len(mode)))
    n_accepted = 0
    for t in range(n_iter):
        log_w_proposal = log_target(proposals[t]) - log_g[t]
        if np.log(rng.uniform()) < log_w_proposal - log_w:  # accept
            phi = proposals[t]
            log_w = log_w_proposal
            n_accepted += 1
        draws[t] = phi
    return draws, n_accepted / n_iter


# ----------------------------------------------------------------------
# Diagnostics: for one quantity, `chains` has shape (number of chains, draws per chain)
# ----------------------------------------------------------------------

# Targets every MCMC run must meet (lecture notes, App. C.3 and C.4, p. 90)
RHAT_MAX = 1.01
ESS_MIN = 1_000
ACCEPT_MIN, ACCEPT_MAX = 0.2, 0.5


def autocorr(x, k):
    """Lag-k autocorrelation rho_k of one chain x."""
    x = x - x.mean()
    return np.sum(x[: len(x) - k] * x[k:]) / np.sum(x * x)


# ======================================================================
# [COURSE] Split-R-hat: Ch. 6, Sec. 6.6, Definition 6.2 (p. 73)
# Split each chain in two halves; compare the variance between the means of the halves with the
# variance within them. If all halves sample the same distribution, R-hat is close to 1.
# ======================================================================
def split_rhat(chains):
    """Split-R-hat of one quantity."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]])  # 2m sequences of length n
    W = halves.var(axis=1, ddof=1).mean()       # average within-sequence variance
    B_over_n = halves.mean(axis=1).var(ddof=1)  # variance of the sequence means, B / n
    return np.sqrt(((n - 1) / n * W + B_over_n) / W)


# ======================================================================
# [COURSE] Effective sample size: Ch. 5, Sec. 5.5, Theorem 5.2 and eq. (5.4) (p. 64)
# N correlated draws are worth ESS = N / tau independent ones, with tau = 1 + 2 sum_k rho_k,
# and the Monte Carlo standard error of a mean is sd / sqrt(ESS) (App. C.1, p. 88).
# ======================================================================
def ess(chains):
    """Effective sample size of all the draws of one quantity, chains pooled."""
    m, n = chains.shape
    # DECISION: the course defines tau but no rule to stop the sum; we stop at the first negative
    # autocorrelation (averaged over chains), beyond which the estimates are mostly noise
    tau = 1.0
    for k in range(1, n):
        rho_k = np.mean([autocorr(chain, k) for chain in chains])
        if rho_k < 0:
            break
        tau += 2 * rho_k
    return m * n / tau


# ======================================================================
# [COURSE] Monte Carlo estimators of posterior summaries: Ch. 1, Sec. 1.4.2, Definition 1.7 (p. 23)
# Posterior means, quantiles and probabilities are averages or quantiles of the draws.
# ======================================================================
def summarize(chains):
    """Posterior mean, sd and 95% interval of one quantity, with its split-R-hat, ESS and MCSE."""
    n_eff = ess(chains)
    return {"mean": chains.mean(),
            "sd": chains.std(ddof=1),
            "2.5%": np.quantile(chains, 0.025),
            "97.5%": np.quantile(chains, 0.975),
            "R-hat": split_rhat(chains),
            "ESS": n_eff,
            "MCSE": chains.std(ddof=1) / np.sqrt(n_eff)}


def convergence_checks(table, accept_rates):
    """PASS (True) or FAIL (False) for each convergence target.

    table: DataFrame with one row per quantity and columns "R-hat" and "ESS" (from summarize)
    accept_rates: acceptance rate of each chain
    """
    accept_rates = np.asarray(accept_rates)
    return {
        f"split-R-hat < {RHAT_MAX} for every quantity": bool((table["R-hat"] < RHAT_MAX).all()),
        f"ESS >= {ESS_MIN:,} for every quantity": bool((table["ESS"] >= ESS_MIN).all()),
        f"acceptance rate in [{ACCEPT_MIN}, {ACCEPT_MAX}] for every chain":
            bool(((accept_rates >= ACCEPT_MIN) & (accept_rates <= ACCEPT_MAX)).all()),
    }


# ======================================================================
# [COURSE] Convergence diagnostics: Ch. 6, Sec. 6.6 (p. 73); autocorrelations: Ch. 5, Example 5.8 (p. 64)
# Chains from dispersed starts should forget them within the burn-in, then look like a stationary
# "hairy caterpillar"; their autocorrelations should die out after a few lags.
# ======================================================================
def plot_diagnostics(draws, labels, truth=None, n_start=500, max_lag=50):
    """Figure with one row per quantity: first iterations, traces after burn-in, autocorrelations.

    draws: array (chains, iterations, quantities), burn-in included; labels: one axis label per quantity
    truth: optional true values, drawn as black lines. Returns the figure.
    """
    n_chains, n_iter, k = draws.shape
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, n_chains))  # one shade of blue per chain
    fig, axes = plt.subplots(k, 3, figsize=(6.3, 1.2 * k + 0.4), sharex="col", squeeze=False)
    for i in range(k):
        for c in range(n_chains):
            chain = draws[c, :, i]
            axes[i, 0].plot(chain[:n_start], color=shades[c], linewidth=0.6)
            axes[i, 1].plot(np.arange(BURN_IN, n_iter), chain[BURN_IN:], color=shades[c],
                            linewidth=0.2, rasterized=True)  # rasterized: a light PDF
            acf = [autocorr(chain[BURN_IN:], lag) for lag in range(max_lag + 1)]
            axes[i, 2].plot(acf, color=shades[c], linewidth=0.8)
        if truth is not None:
            axes[i, 0].axhline(truth[i], color="black", linewidth=0.8)
            axes[i, 1].axhline(truth[i], color="black", linewidth=0.8)
        axes[i, 0].set_ylabel(labels[i])
    axes[0, 0].set_title(f"First {n_start} iterations")
    axes[0, 1].set_title("After burn-in")
    axes[0, 2].set_title("Autocorrelation")
    axes[-1, 0].set_xlabel("Iteration")
    axes[-1, 1].set_xlabel("Iteration")
    axes[-1, 1].locator_params(axis="x", nbins=3)  # few ticks, so the long labels do not overlap
    axes[-1, 2].set_xlabel("Lag")
    return fig


# ----------------------------------------------------------------------
# Risk measures of the one-day loss L = -y
# ----------------------------------------------------------------------

# ======================================================================
# [COURSE] Posterior of a function of the parameters: Ch. 1, Sec. 1.2.2, Definition 1.4 (p. 16)
# VaR and ES are functions h(theta). Applied to every posterior draw, they give draws from the
# posterior of VaR and ES, whose means, intervals and probabilities follow by Monte Carlo.
# ======================================================================
def var_t(theta, alpha):
    """Value-at-Risk at level alpha under the Student-t model: VaR = -mu + sigma q, q the alpha-quantile.

    theta: one vector (mu, sigma, nu) or an array of draws whose last axis is (mu, sigma, nu).
    """
    theta = np.asarray(theta)
    mu, sigma, nu = theta[..., 0], theta[..., 1], theta[..., 2]
    return -mu + sigma * stats.t.ppf(alpha, nu)


def es_t(theta, alpha):
    """Expected Shortfall at level alpha under the Student-t model (finite for nu > 1); theta as in var_t.

    ES = -mu + sigma f(q) / (1 - alpha) (nu + q^2) / (nu - 1), with f the standard Student-t density.
    """
    theta = np.asarray(theta)
    mu, sigma, nu = theta[..., 0], theta[..., 1], theta[..., 2]
    q = stats.t.ppf(alpha, nu)
    return -mu + sigma * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)


def var_gauss(theta, alpha):
    """Value-at-Risk at level alpha under the Gaussian model: -mu + s z_alpha; theta's last axis is (mu, s)."""
    theta = np.asarray(theta)
    return -theta[..., 0] + theta[..., 1] * stats.norm.ppf(alpha)


def es_gauss(theta, alpha):
    """Expected Shortfall at level alpha under the Gaussian model: -mu + s phi(z_alpha) / (1 - alpha)."""
    theta = np.asarray(theta)
    return -theta[..., 0] + theta[..., 1] * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)


def tail_t(theta, v):
    """For each draw of the Student-t model, P(L > v | theta) and E[L 1{L > v} | theta].

    With L = -mu + sigma Z and c = (v + mu) / sigma: P(Z > c) and E[Z 1{Z > c}] = f(c) (nu + c^2) / (nu - 1).
    """
    theta = np.asarray(theta)
    mu, sigma, nu = theta[..., 0], theta[..., 1], theta[..., 2]
    c = (v + mu) / sigma
    prob = stats.t.sf(c, nu)
    mean_beyond = -mu * prob + sigma * stats.t.pdf(c, nu) * (nu + c**2) / (nu - 1)
    return prob, mean_beyond


def tail_gauss(theta, v):
    """For each draw of the Gaussian model, P(L > v | theta) and E[L 1{L > v} | theta]."""
    theta = np.asarray(theta)
    mu, s = theta[..., 0], theta[..., 1]
    c = (v + mu) / s
    prob = stats.norm.sf(c)
    mean_beyond = -mu * prob + s * stats.norm.pdf(c)
    return prob, mean_beyond


# ======================================================================
# [COURSE] Posterior predictive: Ch. 3, Sec. 3.1.2, Example 3.2 (p. 41); Rao-Blackwell: Ch. 3,
# Sec. 3.5.2, Proposition 3.4 (p. 47)
# Tomorrow's loss follows a mixture of the model over the posterior draws. Instead of simulating
# losses, we average over the draws the exact probability and the exact mean beyond a level v:
# the same target, with the noise of simulating the loss removed.
# ======================================================================
def predictive_var_es(theta_draws, alpha, model="student_t"):
    """VaR and ES at level alpha of the posterior predictive loss, a mixture over the posterior draws.

    theta_draws: posterior draws, last axis (mu, sigma, nu) for "student_t" or (mu, s) for "gaussian".
    VaR solves mean_i P(L > VaR | theta_i) = 1 - alpha, and ES = mean_i E[L 1{L > VaR} | theta_i] / (1 - alpha).
    """
    theta = np.asarray(theta_draws).reshape(-1, np.shape(theta_draws)[-1])  # all draws in one list
    tail = tail_t if model == "student_t" else tail_gauss

    def excess_probability(v):
        return tail(theta, v)[0].mean() - (1 - alpha)

    var = brentq(excess_probability, -1, 1)  # a daily log-loss lies well inside (-100%, +100%)
    es = tail(theta, var)[1].mean() / (1 - alpha)
    return var, es
