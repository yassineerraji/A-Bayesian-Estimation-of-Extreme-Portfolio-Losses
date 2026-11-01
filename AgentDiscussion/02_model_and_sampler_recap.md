# 02 · Model, sampler, diagnostics: recap

## Built

| File | Content |
|---|---|
| `src/model.py` | `to_natural` / `to_phi` (scale changes), `sample_prior`, hand-written `log_likelihood`, `log_prior` (with Jacobian), `log_posterior` |
| `src/mh.py` | `rw_metropolis` (Algorithm 4); `run_chains`: BFGS MAP → **3 pilot rounds** → frozen proposal → one chain per starting point, each with its own random stream |
| `src/diagnostics.py` | `autocorr` (FFT), `split_rhat` (Def. 6.2), `ess`, `mcse`, `summarize`, `check_targets` (PASS/FAIL) |
| `src/report.py` | + `plot_diagnostics` (3 columns: first 500 iterations, after burn-in, autocorrelation); long traces rasterized at 300 dpi so the PDF stays small |
| `scripts/03_validate_simulated.py` | prior-only check → simulate → fit → recovery table, PASS/FAIL, figure. **Runs in about 7 s.** |
| `config.toml` | + `[prior]`, `[mcmc]` (including `pilot_rounds`), `[diagnostics]`, `[validation]` |
| `README.md` | run order, outputs and structure updated |

Outputs: `report/tables/prior_check.tex`, `report/tables/validation_recovery.tex`, `report/figures/validation_diagnostics.pdf`.

## Results (printed by the script)

| | True | Mean | 95% interval | R̂ | ESS |
|---|---|---|---|---|---|
| μ | 0.001 | 0.000751 | [0.000512, **0.000993**] | 1.0005 | 6,286 |
| σ | 0.007 | 0.006906 | [0.006651, 0.007162] | 1.0005 | 6,674 |
| ν | 2.5 | 2.380 | [2.202, 2.574] | 1.0007 | 7,130 |

- Acceptance rates: 0.31–0.32.
- **PASS:** R̂, ESS and acceptance.
- **FAIL:** "truth inside the 95% intervals". μ misses by 0.000007.
- **Prior-only check:** MH within about 2–3% of the exact quantiles, so the Jacobian is right.
- **On the figure:** a chain that starts at ν ≈ 75 converges within about 150 iterations, and every chain is stationary well before the 2,000-iteration burn-in.

## Differences from the plan, and why

1. **3 pilot rounds instead of 1** (new `pilot_rounds` setting).
   - BFGS stops with "precision loss", and its curvature estimate can be up to 70 times too small for log(ν−2).
   - With 1 round, **3 of 10 simulated data sets gave an ESS of 13–29**. With 3 rounds, all 10 gave 5,898 or more.
   - Our first run had simply been lucky.
2. **One independent random stream per section** (`spawn(3)`), so changing the sampler settings no longer changes the simulated data set. That change is what produced the data set where μ is missed.
3. **The MAP search starts from the data** (median, log sd, ν = 3), not from the first starting point.

## Checks done beyond the plan

- **Our ESS and R̂ against ArviZ 1.3.0** (installed in a throwaway environment, not in the project), on 3 fits:
  - ESS within 0–8% of ArviZ, always slightly lower, so on the safe side;
  - R̂ identical to 4 decimals.
- **The μ miss is in the data, not the sampler.** SciPy's MLE on the same simulated data gives μ = 0.000763, against our posterior mean of 0.000751.
  - With three parameters, roughly 1 data set in 7 has some 95% interval that misses its true value.
  - **I did not change the seed to make it pass.** Step 3's coverage study measures calibration over about 100 data sets.

## For your review

- **Keep this data set and explain the miss in the report** (my recommendation; the template asks to "explain any discrepancy before moving to the real data"), or present another one? Changing the seed until it passes would be cherry-picking.
- **Every uncertain point is now in `TO_FIX_BEFORE_SUBMISSION.md`**, in a new section, "Method and results to investigate" (9 items).

## Run

```bash
.venv/bin/python code/scripts/03_validate_simulated.py
```

**Next:** once you approve, the Step 3 plan: the validation study (coverage over many simulated data sets, and other ν regimes).
