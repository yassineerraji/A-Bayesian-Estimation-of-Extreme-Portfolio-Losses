# To fix before submission

Concerns to resolve before **Monday 16 November 2026, 23:59 (Paris)**. Tick each one when it's done.

## Environment

- [ ] **Course `environment.yml` not found** (not on EduNao as far as we know). We use `.venv` + a pinned `requirements.txt` for now.
      → Get the file (EduNao, professor or classmates), compare the Python and package versions, rerun every script in that environment, and adapt `requirements.txt`.
- [ ] **Python ≥ 3.11 is required**, because `config.toml` is read with the built-in `tomllib`.
      → Check the course environment's Python version. If it is older, switch the config reader (e.g. to a `config.py`).
- [ ] **Exact reproducibility depends on package versions.** NumPy does not guarantee identical random draws across its versions.
      → Do the final run in the pinned environment and state the versions in the README.

## Data

- [ ] **The source is the Fama–French US market (CRSP value-weighted), not the S&P 500** named in the pitch.
      → Name it correctly in the report and cite the data library (and CRSP).
- [ ] **Licence.** The site states a copyright but no explicit licence.
      → Decide whether shipping the saved data file is acceptable. If not, ship only the download script, which the instructions allow.
- [ ] **The source is revised monthly.**
      → Ship the saved snapshot, and record the download date and CRSP vintage in the README.
- [ ] **The URL could change before grading.**
      → The shipped snapshot is the fallback.

## Method and results to investigate

- [ ] **The single-data-set recovery misses μ narrowly** (Step 2): the 95% interval is [0.000512, 0.000993] against a true 0.001. SciPy's MLE on the same data gives μ = 0.000763, matching our posterior mean of 0.000751, so it's the data, not the sampler.
      → Recommended: keep this seed and explain the miss in the report (the template asks us to "explain any discrepancy"); the Step 3 coverage study gives the calibrated answer. **Do not change the seed to make it pass.**
- [ ] **Tuning robustness.** With 1 pilot round, 3 of 10 simulated data sets gave an ESS below 30. With 3 rounds, all 10 were above 5,000.
      → Watch the coverage-study runs (Step 3) for FAILs. If any appear, raise `pilot_rounds`, or replace BFGS's inverse Hessian with a finite-difference Hessian.
- [ ] **BFGS ends with "precision loss"**, so the MAP is not tightly converged. It's only used as a start, which is harmless.
      → Never report the MAP as a result without a better optimizer.
- [ ] **The ESS stopping rule is our own simplification** (sum up to the first negative autocorrelation). Cross-checked against ArviZ 1.3.0 on 3 fits: our ESS is within 0–8% and always slightly lower (conservative); R̂ is identical to 4 decimals.
      → State the rule and the cross-check in the report (appendix).
- [ ] **The prior-only check is compared by eye** (MH within about 2–3% of the exact quantiles), with no formal tolerance or Monte Carlo error for quantiles.
      → Consider adding an MCSE for quantiles, or a Kolmogorov–Smirnov test.
- [ ] **ν is close to its lower bound**: the data give ν ≈ 2.4–2.5, against `nu_min = 2`.
      → Check the posterior mass near 2. Consider a sensitivity run with `nu_min = 1` (ES only needs ν > 1).
- [ ] **The data table reports the excess kurtosis**, which is infinite under the fitted model (ν < 4).
      → Add a caveat in the report, or drop that row.
- [ ] **MCSE is reported only for posterior means.** The VaR and ES intervals (quantiles) need their own Monte Carlo error (Step 5).
- [ ] **The post-burn-in trace panels are overplotted** (the darkest chain is drawn on top). Minor.
      → If it reads poorly in print, thin the draws for display only.

## Report

- [ ] **Facts in `EXPLANATION.md` that came from general knowledge**: the 1987 crash, the "cubic law" for the tail index, Basel's ES at 97.5%.
      → Verify and cite them if used in the report, otherwise drop them.
- [ ] **Every number in the report comes from the code** (tables via `\input`). Check that nothing is typed by hand.
- [ ] **Template:**
  - remove every `\guide{...}`;
  - underline the name of the member who submits;
  - keep the main text to at most 8 pages;
  - every figure has labelled axes and a caption.

## Submission package

- [ ] `project-erraji-amar.zip` contains **only** `report.pdf`, `report/` and `code/`. No `.venv`, `__pycache__/`, `AgentDiscussion/`, `context/`, `INSTRUCTIONS.md`, `EXPLANATION.md` or this file.
- [ ] A from-scratch run of every script in a clean environment works. The README gives the order of the scripts, and the runtime if it is more than a few minutes.
- [ ] Fixed seeds everywhere.
