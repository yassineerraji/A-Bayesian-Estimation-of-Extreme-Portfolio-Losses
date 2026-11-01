"""Convergence and accuracy diagnostics for several MCMC chains of one quantity, array of shape (chains, draws)."""

import numpy as np


def autocorr(x):
    """Autocorrelation of one chain at every lag, computed with the FFT."""
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    acf = np.fft.irfft(f * np.conj(f))[: len(x)]
    return acf / acf[0]


# ======================================================================
# [COURSE] Split-R̂: Ch. 6, Sec. 6.6, Def. 6.2
# Split each chain in two; compare the variance between the sequence means
# with the variance within sequences. Values above 1.01 signal non-mixing.
# ======================================================================
def split_rhat(chains):
    """Split-R-hat of one quantity."""
    n = chains.shape[1] // 2
    seqs = np.concatenate([chains[:, :n], chains[:, n:2 * n]])
    W = seqs.var(axis=1, ddof=1).mean()      # average within-sequence variance
    B_over_n = seqs.mean(axis=1).var(ddof=1)  # variance of the sequence means
    return np.sqrt(((n - 1) / n * W + B_over_n) / W)


# ======================================================================
# [COURSE] Effective sample size: Ch. 5, Sec. 5.5, Thm 5.2, eq. (5.4)
# ESS = N / tau with tau = 1 + 2 sum_k rho_k; MCSE = sd / sqrt(ESS) (Ch. 6, Sec. 6.6).
# ======================================================================
def ess(chains):
    """Effective sample size of all the draws pooled."""
    rho = np.mean([autocorr(c) for c in chains], axis=0)  # autocorrelation averaged over chains
    # DECISION: the course defines tau but no truncation rule; we sum up to the first negative autocorrelation.
    negative = np.flatnonzero(rho < 0)
    k = negative[0] if negative.size else len(rho)
    return chains.size / (1 + 2 * rho[1:k].sum())


def mcse(chains):
    """Monte Carlo standard error of the posterior mean."""
    return chains.std(ddof=1) / np.sqrt(ess(chains))


# ======================================================================
# [COURSE] Monte Carlo estimator of posterior summaries: Ch. 1, Def. 1.7
# Posterior means, intervals and probabilities are averages over the draws.
# ======================================================================
def summarize(chains):
    """Posterior mean, sd and 95% interval of one quantity, with its split-R-hat, ESS and MCSE."""
    return {"mean": chains.mean(), "sd": chains.std(ddof=1),
            "lo": np.quantile(chains, 0.025), "hi": np.quantile(chains, 0.975),
            "rhat": split_rhat(chains), "ess": ess(chains), "mcse": mcse(chains)}


def check_targets(summaries, accept_rates, targets):
    """PASS/FAIL of the convergence targets ([diagnostics] settings) for a list of summaries."""
    return {
        f"split-R-hat < {targets['rhat_max']}": all(s["rhat"] < targets["rhat_max"] for s in summaries),
        f"ESS >= {targets['ess_min']}": all(s["ess"] >= targets["ess_min"] for s in summaries),
        f"acceptance in [{targets['accept_min']}, {targets['accept_max']}]":
            all(targets["accept_min"] <= a <= targets["accept_max"] for a in accept_rates),
    }
