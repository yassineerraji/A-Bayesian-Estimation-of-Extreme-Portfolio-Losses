# 07 · Report: plan

(Run straight through, as you asked.)

## Goal

`report/report.tex`, built on the course template (same font, margins and spacing), compiled to `report/report.pdf`:
- **at most 8 pages** before the references;
- every number, table and figure produced by the code.

## How numbers get into the text

- **Tables:** `\input{tables/...}` (written by the scripts).
- **Figures:** `\includegraphics{figures/...}`.
- **Numbers inside sentences:** `\val{key}`, defined in the preamble and read from `numbers/*.tex` (written by the scripts). A missing key prints **??key??** in bold, so a typo can't slip through silently.

## Structure (as the instructions suggest)

1. **Introduction:** question, context, what we found.
2. **Data:** source, preprocessing, Table 1, Fig. 1, the features the model must capture.
3. **Model:** DAG (TikZ), likelihood, priors with justification and the prior predictive check, assumptions, VaR/ES, predictive, Gaussian benchmark and 2×2.
4. **Method:** MH with Jacobian (Algorithm 1), tuning, chains and starts, diagnostics, predictive computation (Rao–Blackwell), IS cross-check, what is our own code.
5. **Validation:** prior-only check, one simulated data set (Table 2, Fig. 2, the μ miss explained), coverage study (Table 3).
6. **Results:**
   - 6.1 diagnostics and posterior (Table 4, Fig. 3);
   - 6.2 VaR and ES (Table 5, Fig. 4);
   - 6.3 model checks (Table 6, Fig. 5, Table 7);
   - 6.4 sensitivity and cross-check (Table 8).
7. **Discussion:** the answer in plain words, limitations, next steps.
- **Appendices** (not counted): derivations; extra tables and figures; code map.

## Tool

The LaTeX engine is **Tectonic** (installed). It compiled the course template unchanged.

## Items left to you (`TO_FIX`)

- Institution, program and e-mails in the title block; which member to underline (the one who submits).
- Verify every bibliography entry. They were written from memory; none was looked up.
