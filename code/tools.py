"""Functions shared by our notebooks.

Every notebook imports what it needs from this file (`from tools import ...`), so that each
function is written once. The file grows with the project; its sections are:

    Figures and tables
        COLOR_T, COLOR_GAUSS, COLOR_OTHER, COLOR_DATA   our colours, one per role
        set_style()                                     the figure style of every notebook
        save_figure(fig, name)                          save a figure in ../report/figures/
        save_table(table, name)                         save a table in ../report/tables/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------

# Our colours: the first three colours of the Okabe-Ito palette, which stay distinct for
# colour-blind readers (checked with a colour-vision simulator), plus a neutral grey for data.
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
    })


def save_figure(fig, name):
    """Save a figure as a PDF in ../report/figures/ (the notebook still shows it)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")


def latex_cell(value):
    """Text of one table cell for LaTeX: escape % and &, and give numbers a true minus sign."""
    text = str(value).replace("%", r"\%").replace("&", r"\&")
    if isinstance(value, (int, float, np.integer, np.floating)):
        text = text.replace("-", "$-$")
    return text


def save_table(table, name):
    """Save a pandas DataFrame as a LaTeX table (booktabs style) in ../report/tables/.

    The index becomes the first column. Cells are written exactly as they are displayed,
    so round or format the numbers before calling this function.
    """
    header = [table.index.name or ""] + list(table.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(table.columns) + "}",
             r"\toprule",
             " & ".join(latex_cell(h) for h in header) + r" \\",
             r"\midrule"]
    for label, row in table.iterrows():
        cells = [latex_cell(label)] + [latex_cell(value) for value in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text("\n".join(lines) + "\n")
