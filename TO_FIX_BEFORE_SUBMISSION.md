# To fix before submission

Concerns to resolve before **Monday 16 November 2026, 23:59 (Paris)**. Ticked items are done; the open ones are listed first.

## Open: needs you

- [ ] **Title block of `report/report.tex`.** Replace the two red placeholders (program, institution, country, e-mail of each author) and **underline the name of the member who submits** (`\underline{...}`).
- [ ] **Verify every bibliography entry** in `report/references.bib` (authors, title, journal, volume, pages, year). They were written from memory, and none was looked up: Artzner et al. 1999; BCBS 2019; Mandelbrot 1963; Praetz 1972; Blattberg & Gonedes 1974; Fama & French 1993; Metropolis et al. 1953; Hastings 1970; Roberts et al. 1997; Gelman & Rubin 1992; Vehtari et al. 2021; Flegal & Jones 2010; Gelman, Meng & Stern 1996; Bollerslev 1986; Talts et al. 2018. In particular, check that the Basel standard (BCBS 2019) is the right document for "ES at 97.5%". Also check how the professor wants the lecture notes cited (`leite2026notes`).
- [ ] **Course `environment.yml` not found.** We use a pinned `requirements.txt` (Python 3.12.13, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, matplotlib 3.11.2).
      → If you can get the course file, rerun everything in it and compare the numbers. Python ≥ 3.11 is required (`tomllib`). NumPy does not guarantee identical random draws across its versions.
- [ ] **Data licence.** The Fama–French site states a copyright but no explicit licence.
      → Decide whether to ship `code/data/raw/` in the zip (recommended: it guarantees the exact numbers, since the source is revised monthly). If not, ship only the download script, which the instructions allow. The script will then fetch a newer vintage, and the numbers will move slightly.
- [ ] **Compile check in your own LaTeX setup.** The PDF was built with Tectonic (XeTeX); the main text ends exactly at the bottom of page 8. If you compile on Overleaf or with pdfLaTeX, check that the references still start on page 9. If they don't, a line or two must go.
- [ ] **Build the zip** (from the repository root, after the items above):
      ```bash
      cp report/report.pdf report.pdf
      zip -r project-erraji-amar.zip report.pdf report code -x "*/__pycache__/*" "*.DS_Store"
      ```
      It must contain **only** `report.pdf`, `report/` and `code/`: no `.venv`, `AgentDiscussion/`, `context/`, `INSTRUCTIONS.md`, `EXPLANATION.md` or this file.

## Open: optional improvements

- [ ] **ESS rule cross-check in the code.** Our ESS (sum up to the first negative autocorrelation) matched ArviZ 1.3.0 within 0–8% on 3 fits, but in a throwaway environment, so the report does not claim it. To cite it, add a small optional script using ArviZ (a diagnostics library, not a probabilistic-programming one) and its numbers.
- [ ] **Coverage at the ν = 4 edge (90–91%).** It's explained in the report (a single effect, a mild prior pull). Rerunning with 200 data sets per regime (`[coverage] n_datasets`, about 14 minutes) would narrow the range.
- [ ] **A formal prior-only check** (Monte Carlo error for quantiles, or a Kolmogorov–Smirnov test) instead of "within 2%".
- [ ] **Cosmetic touches.** Model names print as "Student-t" (not italic) inside the tables, and the post-burn-in trace panels are overplotted.

## Done

- [x] **Data:** the source is named and cited (Fama/French data library, CRSP), not "S&P 500"; the vintage (202609 CRSP) and download date are in the README; the snapshot is never overwritten.
- [x] **The single-data-set miss on μ** is kept and explained in the report: SciPy's MLE on the same data agrees with our posterior, so the data are responsible. The coverage study gives the calibrated answer.
- [x] **Tuning robustness:** a finite-difference Laplace covariance, 3 pilot rounds, and starts at the MAP + 5 sd. All 300 coverage fits and all 7 real-data fits converge.
- [x] **BFGS "precision loss":** the MAP is only a starting point and is never reported.
- [x] **The starting points and the ESS rule** are justified and stated in the Method section.
- [x] **Coverage:** lighter settings are stated, the ν = 4 edge is explained, and simulation-based calibration is mentioned as a next step.
- [x] **ν near its bound:** the ν > 1 sensitivity run changes nothing; the report says the interval is clear of 2.
- [x] **Skewness and kurtosis rows** are dropped (infinite when ν < 3); the report says why.
- [x] **The implied sd against the sample sd, and the location against the mean**, are explained in Results 6.1.
- [x] **Every number in the text comes from the code**, through `report/numbers/*.tex` and `\val{...}`. A missing key prints in bold as ??key??; none remain. The typed constants are model or setting inputs only (prior hyperparameters, levels, the 2.38 scaling).
- [x] **Monte Carlo error of quantiles** by batch means (at most 0.08 points), cited.
- [x] **The 99.9% overshoot against history** is discussed as a limitation (the model is conservative there).
- [x] **Plug-in = posterior mean**, stated in the Model section.
- [x] **Template:** no `\guide` text left, figures have labelled axes and captions, the main text ends at the bottom of page 8, and the appendices hold the extra material.
- [x] **From-scratch run in a fresh environment** built from `requirements.txt` (see the Step 7 recap).
- [x] **Fixed seeds everywhere.**
- [x] **`EXPLANATION.md` facts:** the 1987 crash and the "cubic law" are not used in the report. The Basel 97.5% claim is cited (to verify, see above).
