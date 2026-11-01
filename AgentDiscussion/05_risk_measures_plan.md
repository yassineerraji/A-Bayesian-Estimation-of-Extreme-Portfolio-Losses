# 05 · Risk measures: plan

(You asked me to run through to the end, so plan and recap are written without a review stop.)

## Goal

Answer the question: the one-day VaR and ES of the market, with their uncertainty, and how they differ from the Gaussian answer. Split the difference into **heavy tails** and **parameter uncertainty**, using the 2×2 comparison.

## Files

| File | Content |
|---|---|
| `src/risk.py` | + `var_gauss`, `es_gauss`; + `tail_t`, `tail_gauss` (P(Z > c) and E[Z 1{Z > c}] of the standard distribution); + `predictive_var_es` |
| `src/diagnostics.py` | + `batch_se`: Monte Carlo error of any statistic of the draws (batch means) |
| `src/report.py` | + `write_numbers` (numbers quoted in the text, as LaTeX macros); `write_table` gets an optional grouped header |
| `scripts/06_risk_measures.py` | tables `risk_measures.tex` (main) and `risk_mcse.tex` (appendix), figure `risk_measures.pdf`, numbers `numbers/risk.tex` |
| `config.toml` | + `[risk]`: levels (97.5%, 99%, 99.9%), number of batches |

## Decisions

1. **The 2×2 comparison**, for each level:
   - **Plug-in:** VaR and ES at the posterior mean of the parameters. One point estimate, no parameter uncertainty. (The posterior mean equals the MLE to 3 digits here.)
   - **Predictive:** VaR and ES of the posterior predictive loss $L_{T+1}$, a mixture over all 72,000 draws. One number that includes parameter uncertainty.
   - Effect of the tails = t plug-in − Gaussian plug-in. Effect of parameter uncertainty = t predictive − t plug-in.
2. **Posterior of VaR(θ) and ES(θ):** mean and 95% interval, from the draws (closed forms of Step 3). This is the "answer with its uncertainty".
3. **Predictive, computed exactly per draw rather than by simulating returns:**
   - VaR solves $\frac1N\sum_i P_{\theta_i}(L > v) = 1-\alpha$ (root finding, `brentq`);
   - ES $= \frac{1}{1-\alpha}\frac1N\sum_i E_{\theta_i}[L\,1\{L>v\}]$, using the closed form $E[Z\,1\{Z>c\}] = f_\nu(c)\,(\nu+c^2)/(\nu-1)$.
   - This is **Rao–Blackwellization** (Ch. 3, Prop. 3.4): average the exact conditional quantities instead of noisy simulated losses. It matters at 99.9%, where simulation would see very few losses.
4. **Monte Carlo error by batch means:** split each chain into 10 batches of 1,800 draws, far longer than τ ≈ 12, so the batches are nearly independent. MCSE = sd of the batch statistics / √40. It applies to the predictive values and to the interval endpoints (quantiles), which closes the `TO_FIX` item.

## Course flags

| Concept | Where in the notes |
|---|---|
| Rao–Blackwell | Ch. 3, Prop. 3.4 |
| Posterior predictive probability | Ch. 3, Sec. 3.1.2, Example 3.2 |
| Functions of the parameters | Ch. 1, Def. 1.4 |
| MC error with correlated draws | Ch. 5, Thm 5.2 |
