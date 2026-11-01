"""Risk measures: one-day VaR and ES of the market, Student-t against Gaussian, plug-in against predictive.

The 2x2 comparison separates two effects: heavy tails (Student-t plug-in against Gaussian plug-in) and
parameter uncertainty (Student-t predictive against Student-t plug-in). The historical (empirical) values of
the observed losses are shown as a model-free benchmark.

Reads:     config.toml; data/returns.csv; results/posterior_draws.npz (written by 05_fit_real_data.py)
Produces:  report/tables/risk_measures.tex    historical; plug-in, predictive and 95% interval of both models
           report/tables/risk_mcse.tex        Monte Carlo errors of the Student-t values (appendix)
           report/figures/risk_measures.pdf   95% intervals and predictive values of both models, every level
           report/numbers/risk.tex            numbers quoted in the text
Run:       python scripts/06_risk_measures.py   (about 30 s)
"""

# Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.diagnostics import batch_se  # noqa: E402
from src.report import COLORS, WIDTH, save_figure, set_style, write_numbers, write_table  # noqa: E402
from src.risk import es_gauss, es_t, predictive_gauss, predictive_t, var_gauss, var_t  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
LEVELS, N_BATCHES = CFG["risk"]["levels"], CFG["risk"]["n_batches"]
FIGURES, TABLES, RESULTS = CFG["paths"]["figures"], CFG["paths"]["tables"], CFG["paths"]["results"]
NUMBERS = TABLES.parent / "numbers"
MODELS = {"Gaussian": (var_gauss, es_gauss, predictive_gauss), "Student-t": (var_t, es_t, predictive_t)}
SHORT = {"Gaussian": "g", "Student-t": "t"}  # model keys in the numbers file


# %% 1. Load the posterior draws (after burn-in) of both models, and the observed losses
saved = np.load(RESULTS / "posterior_draws.npz")
draws = {"Gaussian": saved["gaussian"], "Student-t": saved["student_t"]}
loss = -100 * load_returns(CFG).to_numpy()  # observed one-day losses, in %

# %% 1b. Historical (model-free) VaR and ES: empirical quantile of the losses, and mean loss beyond it
hist = {}
for a in LEVELS:
    v = np.quantile(loss, a)
    hist["VaR", a], hist["ES", a] = v, loss[loss >= v].mean()
    print(f"Historical {100 * a:g}%: VaR {v:.2f}%, ES {hist['ES', a]:.2f}% ({(loss >= v).sum()} days at or beyond the VaR)")

# %% 2. VaR and ES of each model and level, in %: plug-in, predictive, and posterior mean and 95% interval
# DECISION: plug-in = VaR and ES at the posterior mean of the parameters (one point estimate, no parameter uncertainty).
res, post = {}, {}
for model, (var, es, predictive) in MODELS.items():
    theta = draws[model]
    point = theta.mean(axis=(0, 1))
    for a in LEVELS:
        pred = predictive(theta, a)
        for k, (measure, f) in enumerate([("VaR", var), ("ES", es)]):
            post[model, measure, a] = 100 * f(theta, a)  # posterior draws of VaR(theta) or ES(theta)
            res[model, measure, a] = {"plug": 100 * f(point, a), "pred": 100 * pred[k],
                                      "lo": np.quantile(post[model, measure, a], 0.025),
                                      "hi": np.quantile(post[model, measure, a], 0.975)}

# %% 3. Comparison table and printed summary
cells = lambda r: [f"{r['plug']:.2f}", f"{r['pred']:.2f}", f"[{r['lo']:.2f}, {r['hi']:.2f}]"]  # noqa: E731
rows = [[f"{measure} {100 * a:g}%", f"{hist[measure, a]:.2f}", *cells(res["Gaussian", measure, a]),
         *cells(res["Student-t", measure, a])] for a in LEVELS for measure in ("VaR", "ES")]
write_table(["Measure", "Historical", "Plug-in", "Predictive", "95% interval", "Plug-in", "Predictive", "95% interval"],
            rows, TABLES / "risk_measures.tex",
            top=[r" & & \multicolumn{3}{c}{Gaussian} & \multicolumn{3}{c}{Student-$t$} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"])
print("\nOne-day loss, in % (historical | Gaussian: plug-in, predictive, 95% interval | Student-t: same)")
print("\n".join(f"  {row[0]:10s} " + "  ".join(f"{c:>16s}" for c in row[1:]) for row in rows))

# %% 4. Monte Carlo errors of the Student-t values (batch means)
# ======================================================================
# [COURSE] Monte Carlo error with correlated draws: Ch. 5, Thm 5.2
# Batches much longer than the autocorrelation time tau are nearly independent,
# so the spread of a statistic across batches gives its standard error.
# ======================================================================
theta = draws["Student-t"]
mcse_rows = []
for a in LEVELS:
    for k, (measure, f) in enumerate([("VaR", var_t), ("ES", es_t)]):
        r = res["Student-t", measure, a]
        se_pred = 100 * batch_se(theta, lambda b: predictive_t(b, a)[k], N_BATCHES)
        se_lo, se_hi = (batch_se(post["Student-t", measure, a], lambda b: np.quantile(b, q), N_BATCHES) for q in (0.025, 0.975))
        mcse_rows.append([f"{measure} {100 * a:g}%", f"{r['pred']:.3f}", f"{se_pred:.3f}",
                          f"{r['lo']:.3f}", f"{se_lo:.3f}", f"{r['hi']:.3f}", f"{se_hi:.3f}"])
write_table(["Measure", "Predictive", "MCSE", "2.5% quantile", "MCSE", "97.5% quantile", "MCSE"], mcse_rows,
            TABLES / "risk_mcse.tex")
mcse_max = max(float(x) for row in mcse_rows for x in row[2::2])
print(f"\nLargest Monte Carlo error (Student-t, batch means): {mcse_max:.3f} points of %")

# %% 5. Numbers quoted in the text: every value, plus the two effects of the 2x2 comparison
numbers = {f"{m}{100 * a:g}-{SHORT[model]}-{kind}": f"{value:.2f}"
           for (model, m, a), r in res.items() for kind, value in r.items()}
numbers |= {f"{m}{100 * a:g}-hist": f"{value:.2f}" for (m, a), value in hist.items()}
print("\nEffects (% change): heavy tails = t plug-in vs Gaussian plug-in; parameters = t predictive vs t plug-in")
for m in ("VaR", "ES"):
    for a in LEVELS:
        g, t, key = res["Gaussian", m, a], res["Student-t", m, a], f"{m}{100 * a:g}"
        tails, param = 100 * (t["plug"] / g["plug"] - 1), 100 * (t["pred"] / t["plug"] - 1)
        print(f"  {m} {100 * a:g}%: tails {tails:+.0f}%, parameters {param:+.1f}%")
        numbers[f"{key}-tails"] = f"{tails:.0f}".replace("-", "$-$")
        numbers[f"{key}-param"] = f"{param:.1f}".replace("-", "$-$")
        numbers[f"{key}-under"] = f"{100 * (1 - g['plug'] / t['pred']):.0f}"  # how much the Gaussian plug-in understates
        numbers[f"{key}-ratio"] = f"{t['pred'] / g['plug']:.1f}"  # Student-t predictive / Gaussian plug-in
numbers["risk-mcse-max"], numbers["risk-n-batches"] = f"{mcse_max:.2f}", N_BATCHES
write_numbers(numbers, NUMBERS / "risk.tex")

# %% 6. Figure: for every measure, the 95% interval and predictive value of both models, and the historical value
set_style()
fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
labels = [row[0] for row in rows]
for y, ((m, a), label) in enumerate(zip([(m, a) for a in LEVELS for m in ("VaR", "ES")], labels)):
    for model, dy in [("Gaussian", 0.15), ("Student-t", -0.15)]:
        r = res[model, m, a]
        ax.errorbar(r["pred"], y + dy, xerr=[[r["pred"] - r["lo"]], [r["hi"] - r["pred"]]], fmt="o", ms=4,
                    capsize=2, color=COLORS["gaussian" if model == "Gaussian" else "student_t"],
                    label=f"{model}: predictive, 95% interval" if y == 0 else None)
    ax.plot(hist[m, a], y, "D", ms=4, color=COLORS["data"], label="Historical (observed losses)" if y == 0 else None)
ax.set(xscale="log", xlabel="One-day loss (%), log scale", yticks=range(len(labels)), yticklabels=labels,
       ylim=(len(labels) - 0.5, -0.5))
ax.set_xticks([2, 3, 4, 6, 10, 15, 20], labels=["2", "3", "4", "6", "10", "15", "20"])
ax.legend(loc="upper right")
save_figure(fig, FIGURES / "risk_measures.pdf")
print("\n->", TABLES / "risk_measures.tex,", TABLES / "risk_mcse.tex,", FIGURES / "risk_measures.pdf,", NUMBERS / "risk.tex")
