# 05 · Risk measures: recap

## Built

| File | Content |
|---|---|
| `src/risk.py` | + Gaussian VaR and ES; tail functions; `predictive_var_es` (exact mixture VaR by `brentq`, Rao–Blackwellized ES); `predictive_t`, `predictive_gauss` |
| `src/diagnostics.py` | + `batch_se` (batch-means Monte Carlo error of any statistic) |
| `src/report.py` | `write_table(..., top=...)` for grouped headers; `write_numbers` (LaTeX macros `\num{key}` for numbers quoted in the text) |
| `scripts/06_risk_measures.py` | tables `risk_measures.tex` and `risk_mcse.tex`, figure `risk_measures.pdf`, `report/numbers/risk.tex`. **About 5 s** |
| `config.toml` | + `[risk]` (levels, batches) |

## Results (one-day loss, %)

| | Historical | Gaussian (predictive) | Student-t (predictive) | Student-t 95% interval |
|---|---|---|---|---|
| VaR 97.5% | 2.62 | 2.40 | 2.36 | [2.25, 2.48] |
| ES 97.5% | 4.00 | 2.87 | 4.20 | [3.84, 4.64] |
| VaR 99% | **3.60** | 2.86 | **3.60** | [3.36, 3.87] |
| ES 99% | 5.39 | 3.28 | 6.23 | [5.53, 7.07] |
| VaR 99.9% | 8.09 | 3.81 | 9.55 | [8.31, 11.02] |
| ES 99.9% | 9.70 | 4.16 | 16.23 | [13.29, 19.77] |

- **Heavy tails** (Student-t plug-in against Gaussian plug-in): −2% at VaR 97.5%, **+26% at VaR 99%**, +89% at ES 99%, +149% at VaR 99.9%, **+285% at ES 99.9%**.
- **Parameter uncertainty** (Student-t predictive against plug-in): only **+0.2% to +1.5%**. With 20 years of data, the uncertainty widens the intervals (±20% at 99.9%) but barely shifts the predictive value.
- **Against history:** the Student-t matches the 99% VaR exactly (3.60 against 3.60), and the Gaussian is 21% too low. At 99.9%, the Student-t **overshoots** history, with ES 16.2% against 9.7%. The historical figure rests on only 6 days, though.
- **Monte Carlo error** (batch means): at most 0.08 percentage points.

## Differences from the plan

- **Added the historical benchmark** (column and figure markers): a model-free reference and a first fit check.
- **The figure is a single forest plot on a log scale**, not 4 histograms: the Gaussian's narrow spikes made the Student-t histograms unreadable.
