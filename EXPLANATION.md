# Project explanation: Beyond Gaussian Risk

**Bayesian estimation of extreme one-day losses of a stock index**
Yassine Erraji & Simon Amar · *Bayesian Networks and Markov Models in Action* (Prof. Alessandro Leite), Autumn 2026
**Deadline: Monday 16 November 2026, 23:59 (Paris time)**

---

## 0. The project in five lines

1. We study the **daily returns of a stock index** (e.g. the S&P 500).
2. We want to know **how bad a really bad day can get**: the loss exceeded on only 1 day in 100 (**VaR**), and the average loss on those days (**ES**).
3. The usual shortcut assumes returns follow a bell curve (a **Gaussian**). Real returns have **heavy tails**: extreme days happen far more often than a bell curve allows.
4. We fit a **Student-t** model, whose tail heaviness is a parameter ν learned from the data. We use **Bayesian inference**, so every answer comes with its uncertainty.
5. The answer has no closed form. We approximate it with a **Metropolis–Hastings** sampler that we write ourselves, test it on fake data where we know the truth, then run it on real data.

---

## 1. The question

> **How much risk hides beyond the Gaussian tail?**

1. What are the one-day 99% **VaR** and **ES** of the index, with credible intervals?
2. How different are they from the Gaussian answer?
3. How much of the difference comes from **heavy tails**, and how much from **not knowing the parameters exactly**?

**Why it matters.** Banks and funds set capital and risk limits from these numbers. A model that underestimates the tail holds too little capital for exactly the days that hurt.

**Why the answer is uncertain.** The tail is, by definition, made of rare days. Twenty years of data is about 5,000 days. Only about 50 of them fall in the worst 1%, and about 5 in the worst 0.1%. The shape of the tail has to be inferred from very little information.

### 1.1 Finance vocabulary

| Term | Meaning |
|---|---|
| Price $P_t$ | closing level of the index on day $t$ |
| Log-return $r_t = \log(P_t / P_{t-1})$ | daily change; $r_t = 0.01$ means about +1% |
| Loss $L = -r$ | positive when the index falls |
| **VaR**$_{99}$ (Value-at-Risk) | the loss exceeded on only 1% of days: $\Pr(L > \text{VaR}_{99}) = 0.01$ |
| **ES**$_{99}$ (Expected Shortfall) | the average loss on those 1% worst days: $E[L \mid L \ge \text{VaR}_{99}]$ |

VaR tells you **where** the bad zone starts. ES tells you **how bad** it is inside. ES is always at least as large as VaR, and it reacts more to how heavy the tail is.

### 1.2 Why the Gaussian is not enough

- Under a Gaussian, a 5-standard-deviation drop should happen about **once every 14,000 years**. Stock indices have had many, e.g. 19 October 1987 (−20%, about 20 standard deviations).
- A **Student-t** distribution has tails that decay like a power of $x$ instead of like $e^{-x^2/2}$. Its parameter **ν** (degrees of freedom) controls this: **small ν means heavy tails**, and ν → ∞ gives back the Gaussian. Daily index returns typically give ν around 3–5.

How much it changes the answer depends on the level. Below, a Student-t is compared with a Gaussian of the **same variance** (ratio t / Gaussian, from a quick check):

| Level | VaR (ν=4) | ES (ν=4) | VaR (ν=3) | ES (ν=3) |
|---|---|---|---|---|
| 97.5% | 1.00 | 1.21 | 0.94 | 1.24 |
| 99%   | 1.14 | 1.38 | 1.12 | 1.51 |
| 99.9% | 1.64 | 2.03 | 1.91 | 2.64 |

At 99%, VaR barely moves, ES moves by 40–50%, and at 99.9% both roughly double. **This is why we report ES and several levels, not just the 99% VaR.**

---

## 2. Course background, in plain words

The course has six chapters. We rely mostly on Chapters 1, 5 and 6.

### 2.1 Bayesian inference (Chapter 1)

- Unknown parameters $\theta$ are treated as **random**: we describe our beliefs about them with probability distributions.
- **Prior** $\pi(\theta)$: what we believe *before* seeing the data.
- **Likelihood** $p(y \mid \theta)$: how probable the observed data are *if* $\theta$ were the truth.
- **Posterior** $\pi(\theta \mid y)$: what we believe *after* seeing the data. By **Bayes' theorem**:

$$\pi(\theta \mid y) = \frac{p(y \mid \theta)\,\pi(\theta)}{p(y)} \;\propto\; p(y \mid \theta)\,\pi(\theta).$$

- The denominator $p(y) = \int p(y\mid\theta)\pi(\theta)\,d\theta$ (the **evidence**) is an integral we usually **cannot compute**. We only know the posterior "up to a constant".
- The output is a **distribution, not one number**. From it we get a mean, a **95% credible interval** (an interval that contains $\theta$ with probability 0.95), and probabilities of events.
- **Why Bayesian here:** the uncertainty about ν (the tail) flows directly into the uncertainty about VaR and ES.

### 2.2 Bayesian network (Chapter 1)

A **Bayesian network** is a diagram of the model: circles are random quantities, arrows mean "depends on", the shaded circle is observed, and a box (a *plate*) means "repeated for every day". Ours:

```
   μ (location)   σ (scale)   ν (tail)
         \            |           /
          \           |          /
       ┌───▼──────────▼─────────▼───┐
       │            r_t             │   (observed)
       │                 t = 1..T   │
       └────────────────────────────┘
```

### 2.3 Monte Carlo (Chapters 1 and 3)

- Every posterior summary is an integral, e.g. a posterior mean $E[h(\theta)\mid y] = \int h(\theta)\,\pi(\theta\mid y)\,d\theta$.
- **Monte Carlo** replaces the integral by an **average over random draws** $\theta^{(1)},\dots,\theta^{(N)}$ from the posterior: $\frac1N \sum_i h(\theta^{(i)})$. A probability becomes "the fraction of draws where the event happens".
- The error shrinks like $1/\sqrt{N}$, whatever the number of parameters. The **Monte Carlo standard error** $\hat\sigma_h / \sqrt N$ tells us how many digits to trust.
- **The catch:** we need draws from the posterior, and we can't sample it directly. Markov chains solve this (2.6–2.7).
- **Practical rule:** always compute **log** densities. A product of 5,000 densities is too small for a computer and rounds to 0 ("underflow").

### 2.4 Random variable generation (Chapter 2)

- Computers produce **pseudo-random** numbers from a **seed**. The same seed gives the same numbers, which is how we make results reproducible.
- **Change of variables:** if we work with a transformed parameter (e.g. $\log\sigma$ instead of $\sigma$), its density gets multiplied by a correction factor, the **Jacobian**. We need this in 5.2.
- We use NumPy to draw standard random variables (allowed by the rules).

### 2.5 Monte Carlo optimization (Chapter 4), minor role

- The **MAP** is the single most probable parameter value (the peak of the posterior), found with an optimizer.
- We only use it as a **good starting point** for the sampler and to get a **first idea of the posterior's width** for tuning.

### 2.6 Markov chains (Chapter 5)

- A **Markov chain** is a random sequence where the next value depends **only on the current one**.
- A well-behaved chain **forgets where it started** and settles into a fixed **stationary distribution**.
- **Ergodic theorem:** the average along one long run of the chain converges to the average under that stationary distribution. So if we build a chain whose stationary distribution **is our posterior**, its running averages are posterior summaries. That is **MCMC** (Markov chain Monte Carlo).
- **The price:** consecutive draws are **correlated**, so $N$ draws are worth fewer independent ones. That number is the **effective sample size**, $\text{ESS} = N/\tau$, where $\tau = 1 + 2\sum_k \rho_k$ adds up the autocorrelations $\rho_k$ at each lag $k$.
- **Detailed balance**, $\pi(x)K(x,y) = \pi(y)K(y,x)$ (with $K$ the probability of moving from $x$ to $y$), guarantees that $\pi$ is stationary. It only involves **ratios** of $\pi$, so the unknown constant $p(y)$ cancels out.

### 2.7 Metropolis–Hastings (Chapter 6): the core of our project

A recipe for building a Markov chain whose stationary distribution is the posterior. **Random-walk** version:

1. From the current point $\theta$, **propose** a nearby point $\theta' = \theta + \text{random step}$.
2. Compute the **acceptance probability** $a = \min\{1,\ \tilde\pi(\theta')/\tilde\pi(\theta)\}$, where $\tilde\pi$ is the unnormalized posterior (prior × likelihood). The unknown constant cancels.
3. With probability $a$, **move** to $\theta'$. Otherwise **stay** at $\theta$, and record $\theta$ again. Repeats are part of the output.

Moves towards higher posterior density are always accepted, and moves away only sometimes. The chain ends up spending time in each region in proportion to its posterior probability.

**Tuning the step size:**
- Too small, and the chain crawls.
- Too large, and almost every proposal is rejected.
- The rule of thumb is an **acceptance rate of 20–50%**. With correlated parameters, the step should follow the posterior's shape: a covariance of $(2.38^2/d)\,\hat\Sigma$, where $\hat\Sigma$ is estimated from a short **pilot run** and $d$ is the number of parameters.

**Constraints:** a random walk works best on the whole real line. So we sample $\log\sigma$ instead of $\sigma > 0$, and add the Jacobian (2.4).

### 2.8 Convergence diagnostics (Chapter 6)

No test *proves* that a chain has converged. These tools *detect* failures, and we use them together:

- **Trace plot:** each parameter plotted against the iteration. A good chain looks like a "hairy caterpillar", with no trend and no long flat stretches.
- **Burn-in:** we discard the first iterations, while the chain travels from its starting point to the posterior.
- **Several chains (4) from dispersed starts:** if they agree, it is unlikely they are all stuck in the same wrong place.
- **Split-R̂:** compares the variation *between* chains with the variation *within* them. A value near 1 is good, and **above 1.01 is an alarm**.
- **Autocorrelation, ESS, MCSE:** how many effectively independent draws we have, and the resulting Monte Carlo error $\hat\sigma/\sqrt{\text{ESS}}$.
- **Acceptance rate:** checks the tuning.

### 2.9 Checking the model (required by the instructions)

- **Validation on simulated data:** generate fake data from our model with "true" parameters **we choose**, run the sampler, and check that the true values fall inside the 95% credible intervals. It is the only way to tell a **bug** from a **surprising result**.
- **Posterior predictive check:** simulate many fake data sets from the fitted model and compare them with the real data on chosen statistics. If the real data look unusual among the fakes, the model is missing something.
- **Prior sensitivity:** rerun with other reasonable priors. Conclusions should not hinge on one arbitrary choice.

### 2.10 Course tools we do *not* use

Accept–reject (Ch. 2), Gibbs sampling, simulated annealing, EM (Ch. 4–6) and ABC. Importance sampling (Ch. 3) might serve as an optional cross-check.

---

## 3. The model

### 3.1 Data

$r_1, \dots, r_T$: daily log-returns of a broad equity index. Source and period are still open (§9). Twenty years gives $T \approx 5{,}000$.

### 3.2 Unknowns $\theta = (\mu, \sigma, \nu)$

| Symbol | Name | Meaning |
|---|---|---|
| $\mu \in \mathbb R$ | location | the center of daily returns |
| $\sigma > 0$ | scale | the spread. **Not** the standard deviation, which is $\sigma\sqrt{\nu/(\nu-2)}$ |
| $\nu > 2$ | degrees of freedom | tail heaviness. $\nu > 2$ guarantees a finite variance |

### 3.3 Likelihood

Given $\theta$, the days are **independent** and each follows a Student-t:

$$p(r_t \mid \mu,\sigma,\nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\,\sigma}\left[1 + \frac1\nu\left(\frac{r_t-\mu}{\sigma}\right)^2\right]^{-\frac{\nu+1}{2}},
\qquad
\ell(\theta) = \sum_{t=1}^T \log p(r_t \mid \theta).$$

### 3.4 Priors (independent)

| Parameter | Prior | What it says | Why |
|---|---|---|---|
| $\mu$ | $\mathcal N(0,\,0.001^2)$ *(pitch: $0.01^2$, see §9)* | daily drift within ±0.2% a day (about ±50% a year) with 95% probability | rules out absurd drifts, yet is far wider than what the data pin down |
| $\sigma$ | HalfNormal(0.02) | a Gaussian folded onto positive values; 95% of its mass below a 4% daily scale | covers both calm and crisis markets |
| $\nu - 2$ | Exponential, **rate** 0.1 (mean 10) | prior median ν ≈ 9; $\Pr(\nu<5) \approx 26\%$ | lets the data choose between very heavy and near-Gaussian tails |

Write "**rate** 0.1" explicitly: NumPy and SciPy parametrize the exponential by its *scale* (= 10 here). Mixing the two up is a classic bug.

With about 5,000 days, the data dominate all three priors. The sensitivity analysis should confirm it.

### 3.5 Posterior (up to a constant)

$$\pi(\mu,\sigma,\nu \mid r_{1:T}) \;\propto\; \prod_{t=1}^T p(r_t \mid \mu,\sigma,\nu)\;\cdot\;\pi(\mu)\,\pi(\sigma)\,\pi(\nu).$$

No standard distribution has this shape, which is why we need MCMC.

### 3.6 Assumptions, to be stated in the report

- **Independent days.** This ignores **volatility clustering** (calm and turbulent periods). It is the main limitation: our VaR describes *a typical day over the window*, not *tomorrow given today's market*.
- **Same distribution over the whole window.**
- **Symmetric tails.** The Student-t is symmetric, while real crashes tend to be larger than rallies.
- **$\nu > 2$** (finite variance). This is consistent with the finance literature on stock-return tails.

---

## 4. What we compute from the posterior

### 4.1 Risk measures as functions of the parameters

For a Student-t, VaR and ES have formulas. Here $\alpha = 0.99$, $q = $ the $\alpha$-quantile of a standard Student-t with ν degrees of freedom, and $f_\nu$ is its density:

$$\text{VaR}_\alpha(\theta) = -\mu + \sigma q, \qquad
\text{ES}_\alpha(\theta) = -\mu + \sigma\,\frac{f_\nu(q)}{1-\alpha}\cdot\frac{\nu + q^2}{\nu - 1}.$$

The Gaussian versions, with mean $m$ and standard deviation $s$: $\text{VaR} = -m + s\,z_\alpha$ and $\text{ES} = -m + s\,\varphi(z_\alpha)/(1-\alpha)$.

**For each posterior draw** $\theta^{(i)}$ we compute $\text{VaR}(\theta^{(i)})$. This gives a whole **posterior distribution of VaR**, summarized by its mean and 95% credible interval. Same for ES.

### 4.2 Predictive risk (one number that includes parameter uncertainty)

The distribution of **tomorrow's** return, averaged over our uncertainty about $\theta$:
$p(r_{T+1}\mid r_{1:T}) = \int p(r_{T+1}\mid\theta)\,\pi(\theta\mid r_{1:T})\,d\theta$.

We simulate from it by taking a posterior draw $\theta^{(i)}$, then drawing a return from the Student-t with those parameters. Its VaR and ES fold the parameter uncertainty into a single number.

### 4.3 Levels

- **99%**, the headline.
- **ES 97.5%**, the regulatory standard for market risk under Basel.
- **99.9%**, the extreme level, where the uncertainty about ν matters most.

### 4.4 Separating the two effects (the 2×2 comparison)

| | Plug-in (one best guess of θ) | Bayesian (full posterior) |
|---|---|---|
| **Gaussian** | the textbook answer | + parameter uncertainty |
| **Student-t** | + heavy tails | both (**our model**) |

The Bayesian Gaussian model reuses the same Metropolis–Hastings code, with two parameters.

---

## 5. The method, step by step

1. **Data.** Download prices, compute log-returns, check for gaps and errors.
2. **Log-posterior on an unconstrained scale.** We sample $\phi = (\mu,\ \eta,\ \lambda)$ with $\eta = \log\sigma$ and $\lambda = \log(\nu-2)$, so that every value of $\phi$ is valid:
   $$\log\tilde\pi(\phi) = \ell(\mu, e^\eta, 2+e^\lambda) + \log\pi(\mu) + \log\pi_\sigma(e^\eta) + \log\pi_\nu(2+e^\lambda) + \underbrace{\eta + \lambda}_{\text{Jacobian}}.$$
   Forgetting the Jacobian does not crash anything. It **silently biases** the results, and step 7 is what catches it.
3. **MAP** with an optimizer, giving a starting point and a rough posterior width.
4. **Pilot run**, then estimate the posterior covariance $\hat\Sigma$ and **freeze** the proposal $\mathcal N(0,\ (2.38^2/3)\,\hat\Sigma)$.
5. **Main run:** 4 chains from dispersed starting points, with burn-in discarded:
   ```
   φ ← start
   repeat N times:
       φ' ← φ + L·ε,   ε ~ N(0, I₃),   L·Lᵀ = (2.38²/3) Σ̂
       draw U ~ Uniform(0,1)
       if log U < log π̃(φ') − log π̃(φ):  φ ← φ'      # accept
       store φ                                          # also when rejected
   ```
6. **Diagnostics:** traces, split-R̂, autocorrelation, ESS, MCSE, acceptance rates.
7. **Validation on simulated data:**
   - simulate $T$ days from known parameters, e.g. $(\mu,\sigma,\nu) = (0.0003,\ 0.007,\ 4)$, plus a lighter-tailed case $\nu = 8$;
   - check that the truth falls inside the credible intervals;
   - optionally, repeat many times and check that 95% intervals contain the truth about 95% of the time (*coverage*).
8. **Real data:** posterior summaries, VaR and ES, and the 2×2 comparison.
9. **Posterior predictive checks:** compare the real data with simulated data sets on:
   - kurtosis;
   - worst day;
   - number of days beyond 4 standard deviations;
   - empirical 1% quantile;
   - autocorrelation of $r_t^2$. We **expect this one to fail**, which shows the volatility-clustering limitation honestly.
10. **Sensitivity:** other priors on ν (e.g. rate 0.05 and 0.5), a wider prior on μ, and possibly another data window.
11. **Optional cross-check** with PyMC or Stan, which the rules allow *only* for checking.

---

## 6. What we will build (proposed layout, to confirm before coding)

```
project-erraji-amar.zip
├── report.pdf                  # ≤ 8 pages + references + appendices
├── report/                     # LaTeX sources (course template, references.bib, figures/)
└── code/
    ├── README.md               # how to run; which file produces which figure/table
    ├── requirements.txt
    ├── data/download_data.py   # fetches prices (or the data itself if redistributable)
    ├── src/
    │   ├── model.py            # log-likelihood, priors, log-posterior on (μ, log σ, log(ν−2))
    │   ├── mh.py               # random-walk Metropolis–Hastings + pilot tuning
    │   ├── diagnostics.py      # split-R̂, autocorrelation, ESS, MCSE
    │   └── risk.py             # VaR/ES formulas (t and Gaussian), predictive simulation
    └── scripts/ (or notebooks/)
        ├── 01_validation_simulated.py
        ├── 02_fit_real_data.py
        ├── 03_checks_and_sensitivity.py
        └── 04_figures_tables.py
```

**Rules for the code:**
- we write the sampler ourselves;
- fixed random seeds;
- it runs end to end without manual steps;
- every number in the report comes from this code.

---

## 7. Challenges and how we handle them

| Challenge | Why it happens | What we do |
|---|---|---|
| **Slow mixing** | σ and ν trade off: a heavier tail (smaller ν) can be offset by a smaller scale, so the two are correlated | log scales + proposal covariance from a pilot run; check ESS and R̂ |
| **Constraints** (σ > 0, ν > 2) | a random walk can step outside | sample on log scales and add the Jacobian |
| **Silent bugs** | a wrong Jacobian or a rate/scale mix-up still "runs" | validation on simulated data with known truth |
| **Underflow** | product of thousands of densities rounds to 0 | log densities everywhere |
| **Monte Carlo error on ES / 99.9%** | these depend on the rarely-seen tail | report MCSE next to every estimate |
| **Model too simple** | real volatility comes in waves | posterior predictive checks reveal it; stated as the main limitation |
| **Choice of data window** | including crises (2008, 2020) changes ν | justify the choice; possibly test another window |
| **Page limit** | 8 pages go fast | extra diagnostics go in the appendix |

---

## 8. Grading checklist

| Criterion (weight) | What graders look for | Where we cover it |
|---|---|---|
| Modeling (25%) | clear question, every symbol defined, priors justified, assumptions stated | §1, §3 |
| Implementation (25%) | correct method written by us, sensible tuning, readable and reproducible code | §5, §6 |
| Results & diagnostics (30%) | simulated validation, diagnostics, posterior summaries with uncertainty, predictive checks, sensitivity, an answer | §4, §5 (steps 6–10) |
| Report (20%) | structure, figures with labeled axes and captions, honest limitations, page limit | report |

---

## 9. Open decisions

1. **Data source.** Options: FRED (only about the last 10 years of the S&P 500), Stooq, or Yahoo Finance via `yfinance`. Check the licence. A download script avoids the redistribution issue.
2. **Index and window.** E.g. S&P 500 from 2005 to 2024 (about 5,000 days, including 2008 and 2020).
3. **Priors.** Confirm $\mu \sim \mathcal N(0, 0.001^2)$ instead of the pitch's $0.01^2$. Keep the others.
4. **Levels reported.** 99% headline, plus ES 97.5% and 99.9%.
5. **Notebooks or scripts.**

