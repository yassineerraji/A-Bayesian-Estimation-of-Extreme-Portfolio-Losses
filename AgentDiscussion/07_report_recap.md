# 07 · Report: recap

## Built

- **`report/report.tex`**, on the course template: same class, font size, margins and spacing; template packages kept. Packages added:
  - `tikz` for the Bayesian-network diagram;
  - `caption` for the space between a table caption and its table.

  Plus `breaklinks` and an `\emergencystretch` so that no line overflows the margin.
- **`report/references.bib`:** the template's two references plus 19 more (data library, finance, MCMC, diagnostics, the course notes).
- **`report/report.pdf`:** 16 pages in total, laid out as follows:
  - **main text pages 1–8**, ending exactly at the bottom of page 8;
  - references on pages 9–10;
  - appendices on pages 11–16 (derivations, extra tables and figures, code map).
- **Every number in the text comes from the code.** Scripts 02–09 write `report/numbers/*.tex`, and the text uses `\val{key}`. A missing key would print **??key??**; none remain. Tables come in via `\input{tables/...}`, figures from `figures/`.
- **Code changes for the report:**
  - every script writes its numbers;
  - `write_table` typesets real minus signs;
  - the figures were made shorter (trace grids, data overview, risk forest plot);
  - the sensitivity script computes the largest prior-induced change itself (0.6%).

## How the 8-page limit was met (the first draft had 10)

- **Moved to Appendix B:** the data summary table, the two trace figures, the posterior histograms, the predictive-check figure and the sensitivity table. The text keeps their key numbers and points to them.
- **Main-text floats:** data overview figure, Bayesian-network diagram, Algorithm 1, recovery table, coverage table, posterior table, risk table and figure, predictive-check table, exceedances table.
- **Wording tightened throughout.** Every typed number that the code didn't produce (from my exploration notes) was removed or replaced by `\val{...}`.

## Final reproducibility run

All generated outputs were deleted, except the raw data snapshot. The 9 scripts were then run in order in a **fresh environment built from `requirements.txt`** (Python 3.12, the pinned versions).

- **Result: all 9 scripts ran without error in 7 min 48 s.**
- **Every regenerated table (`report/tables/`) and every quoted number (`report/numbers/`) is byte-identical** to the previous outputs.
- The report, recompiled from these files, still ends its main text at the bottom of page 8, with no missing number.

## Left to you

See `TO_FIX_BEFORE_SUBMISSION.md`, section "Open: needs you":
- the title-block placeholders and the underline;
- verifying the bibliography;
- the course environment;
- the data licence;
- compiling in your LaTeX setup;
- building the zip.
