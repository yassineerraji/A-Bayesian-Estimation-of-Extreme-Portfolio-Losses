# 02 · Model, sampler, diagnostics: plan

## Goal

The core of the project, written once in `src/` and reused by every later step:
- the log-posterior;
- our random-walk Metropolis–Hastings;
- the convergence diagnostics.

Then a first full fit on **simulated data with known parameters**, of the same size as the real data (5,031 days).

## New and changed files

| File | Content |
|---|---|
| `src/model.py` | `to_natural(φ)`, `log_likelihood(φ, r)`, `log_prior(φ, prior)` (Jacobian included), `log_posterior(φ, r, prior)` |
| `src/mh.py` | `rw_metropolis(...)` (Algorithm 4); `run_chains(...)`: MAP → pilot → frozen proposal → 4 chains |
| `src/diagnostics.py` | `autocorr`, `split_rhat`, `ess`, `mcse`, `summarize` (mean, sd, 95% interval, R̂, ESS, MCSE) |
| `src/report.py` | + `plot_diagnostics(...)`, the trace/autocorrelation figure reused for the real data in Step 4 |
| `scripts/03_validate_simulated.py` | prior-only check, simulate, fit, diagnostics, recovery table and figure |
| `config.toml` | + `[prior]`, `[mcmc]`, `[diagnostics]`, `[validation]` |

## Model on the sampling scale

We sample $\phi = (\mu, \eta, \lambda)$ with $\sigma = e^\eta$ and $\nu = \nu_{\min} + e^\lambda$, where $\nu_{\min} = 2$. Constants are dropped because they cancel in the acceptance ratio.

$$\log\tilde\pi(\phi) = \underbrace{\textstyle\sum_t \log t_\nu(r_t;\mu,\sigma)}_{\text{likelihood}} \;-\; \frac{\mu^2}{2s_\mu^2} \;-\; \frac{\sigma^2}{2s_\sigma^2} \;-\; \rho\,(\nu-\nu_{\min}) \;+\; \underbrace{\eta + \lambda}_{\text{Jacobian}}$$

with $s_\mu = 0.001$, $s_\sigma = 0.02$ and $\rho = 0.1$, all read from the config.

## Decisions to review

1. **True values $(\mu, \sigma, \nu) = (0.001,\ 0.007,\ 2.5)$**, the rounded maximum-likelihood fit on our real data (ν̂ = 2.47).
   - Why: it tests the sampler where the real posterior lives (very heavy tails, close to the ν > 2 bound).
   - The other regimes (ν = 4 and 8) and the coverage study come in Step 3.
2. **A hand-written Student-t log-density** (with `scipy.special.gammaln`), not `scipy.stats.t.logpdf`.
   - Why: it is the report's equation written as code, and it is 9 times faster (32 µs against 291 µs per evaluation).
   - An `assert` checks that it matches SciPy.
3. **The tuning recipe is the one in Ch. 6, §6.5**, in three stages:
   - find the MAP with BFGS; its inverse Hessian (the Laplace approximation) gives a first proposal;
   - run a 5,000-iteration pilot and estimate the posterior covariance $\hat\Sigma$;
   - freeze the proposal at $\mathcal N(0, (2.38^2/3)\hat\Sigma)$ for the main run.
4. **The 4 starting points are drawn from the prior**, which spreads them far apart by construction (μ is about 8 posterior sd away, σ and ν much further).
   - Each chain runs 20,000 iterations, of which the first 2,000 are discarded as burn-in.
   - We check the burn-in on the traces and with R̂.
5. **ESS.** $\tau = 1 + 2\sum_k \bar\rho_k$, where $\bar\rho_k$ is the autocorrelation averaged over the 4 chains. The sum stops at the first negative $\bar\rho_k$.
   - The course defines τ (Thm 5.2) but gives no rule for where to stop the sum. This is the simplest standard one.
6. **Split-R̂** is exactly Def. 6.2. **MCSE** = sd/√ESS.
7. **Prior-only check: the same sampler run with no data must reproduce the known prior quantiles.**
   - Why: with 5,031 days the data swamp the prior. A **wrong Jacobian would barely move the posterior, and parameter recovery would not catch it.** This check tests the prior and Jacobian code on its own.
8. **Diagnostics and summaries are reported on the natural scale (μ, σ, ν)**, the quantities we interpret.
9. **Targets live in the config** (R̂ < 1.01, ESS ≥ 1,000, acceptance 0.2–0.5). The script prints PASS or FAIL for each.
10. **Diagnostics figure: a 3×3 grid**, one row each for μ, σ and ν. The columns:
    - the first 500 iterations from the dispersed starts, showing the burn-in;
    - the traces after burn-in, with the true value as a line;
    - the autocorrelation up to lag 50.

    The chains are drawn in four shades of the Student-t blue, since chain identity carries no meaning and lightness tells them apart.
11. **Seeds.** One seed in the config; each chain gets its own independent stream via `rng.spawn`.

## Config additions

```toml
[prior]        mu_sd = 0.001, sigma_scale = 0.02, nu_min = 2.0, nu_rate = 0.1
[mcmc]         seed = 2026, chains = 4, pilot_iter = 5000, n_iter = 20000, burn_in = 2000
[diagnostics]  rhat_max = 1.01, ess_min = 1000, accept_min = 0.2, accept_max = 0.5
[validation]   mu = 0.001, sigma = 0.007, nu = 2.5
```

## Outputs of `03_validate_simulated.py` (about 10 s)

- `report/figures/validation_diagnostics.pdf`: the 3×3 grid.
- `report/tables/validation_recovery.tex`: for each parameter, the true value, posterior mean, 95% interval, R̂, ESS and MCSE.
- `report/tables/prior_check.tex` (appendix): 5%, 50% and 95% quantiles, MH against exact.
- Printed: acceptance rates and the PASS/FAIL targets.

## Course flags planned

| Concept | Where in the notes |
|---|---|
| Unnormalized posterior | Ch. 1, §1.2.2, Def. 1.3, eq. (1.6) |
| Log scale against underflow | Ch. 1, Remark 1.3 |
| Change of variables (Jacobian) | Ch. 2, Prop. 2.1; Ch. 6, §6.2 |
| MAP | Ch. 4, Def. 4.1 |
| Random-walk MH | Ch. 6, Algorithm 4, §6.3 |
| Tuning | Ch. 6, §6.5 |
| ESS and τ | Ch. 5, Thm 5.2, eq. (5.4) |
| Split-R̂ and the other diagnostics | Ch. 6, Def. 6.2, §6.6 |
| Monte Carlo summaries | Ch. 1, Def. 1.7 |

## Heads-up from the fit

With ν ≈ 2.5, the model says **the 4th moment is infinite**. Two consequences:
- The data table's "excess kurtosis" is not a stable statistic, so we should not use it as a predictive check (Step 6).
- Tails are heavier than the ν = 3–4 examples in `EXPLANATION.md`, so ES and the 99.9% level will matter even more.

## Not in this step

Real data, the Gaussian model and the coverage study: Steps 3 and 4.
