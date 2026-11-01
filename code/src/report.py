"""Shared helpers for the report's outputs: one figure style and LaTeX tables."""

import re

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics import autocorr

# Fixed color per role, reused in every figure (blue/orange pass the colorblind checks; data in neutral gray)
COLORS = {"data": "#52514e", "student_t": "#2a78d6", "gaussian": "#eb6834",
          "miss": "#d03b3b"}  # status red, only for intervals that miss the truth (always with a legend)
WIDTH = 16 / 2.54  # report text width (16 cm) in inches, so fonts print at their true size
# DECISION: returns-scale quantities (mu, sigma, sd, VaR, ES) are displayed in % in every table and figure
PARAM_LABELS = [r"$\mu$ (%)", r"$\sigma$ (%)", r"$\nu$"]
PARAM_SCALE = np.array([100, 100, 1])  # multiply (mu, sigma, nu) by this to display them


def set_style():
    """Apply the report's figure style: small fonts, hairline grid, no top/right frame."""
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,  # automatic spacing between panels
        "savefig.dpi": 300,  # resolution of rasterized layers (long traces), the rest stays vector
    })


def save_figure(fig, path):
    """Save a figure as PDF (creating its folder if needed) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# [COURSE] Convergence diagnostics: Ch. 6, Sec. 6.6 (autocorrelation: Ch. 5, Sec. 5.5)
# Traces from dispersed starts, burn-in, "hairy caterpillar" after it,
# and autocorrelations that decay fast enough.
# ======================================================================
def plot_diagnostics(draws, labels, burn_in, truth=None, n_start=500, max_lag=50):
    """Figure with one row per quantity: first iterations, traces after burn-in, autocorrelation.

    draws: (chains, n_iter, k) with burn-in included; labels: k axis labels;
    truth: optional k true values, drawn as horizontal lines. Returns the figure.
    """
    set_style()
    k = draws.shape[2]
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, draws.shape[0]))  # one shade of the Student-t blue per chain
    fig, axes = plt.subplots(k, 3, figsize=(WIDTH, 1.1 * k), sharex="col")
    for i, label in enumerate(labels):
        for chain, color in zip(draws[:, :, i], shades):
            axes[i, 0].plot(chain[:n_start], color=color, lw=0.6)
            axes[i, 1].plot(np.arange(burn_in, len(chain)), chain[burn_in:], color=color, lw=0.2, rasterized=True)
            axes[i, 2].plot(autocorr(chain[burn_in:])[: max_lag + 1], color=color, lw=0.8)
        if truth is not None:
            for ax in axes[i, :2]:
                ax.axhline(truth[i], color="#0b0b0b", lw=0.8)
        axes[i, 0].set_ylabel(label)
    for ax, title in zip(axes[0], [f"First {n_start} iterations", "After burn-in", "Autocorrelation"]):
        ax.set_title(title)
    for ax, xlabel in zip(axes[-1], ["Iteration", "Iteration", "Lag"]):
        ax.set_xlabel(xlabel)
    return fig


def write_table(header, rows, path, top=()):
    """Write a LaTeX booktabs tabular; the caption is written in the report itself.

    header: column names; rows: lists of formatted cells; path: output .tex file;
    top: optional raw LaTeX lines above the header (e.g. column groups with \\multicolumn).
    The first column is left-aligned, the others right-aligned; "%" is escaped and minus signs are typeset.
    """
    def line(cells):  # a hyphen before a digit becomes a real minus sign; "--" ranges are kept
        return " & ".join(re.sub(r"(?<![-\w])-(?=\d)", "$-$", str(c)).replace("%", r"\%") for c in cells) + r" \\"

    cols = "l" + "r" * (len(header) - 1)
    tex = [r"\begin{tabular}{@{}" + cols + r"@{}}", r"\toprule", *top, line(header), r"\midrule",
           *map(line, rows), r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tex) + "\n")


def write_numbers(numbers, path):
    r"""Write the numbers quoted in the report's text as LaTeX macros, used there as \val{key}.

    numbers: dict key -> formatted value; path: output .tex file, read by report.tex with \input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n" for k, v in numbers.items()))
