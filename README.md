# Reference-aware cross-cell SOH evaluation — analysis code and derived data

Package version 1.0 (corresponds to manuscript version v34, 2026-10-22).

Companion code and data for the manuscript

> **Do Partial Incremental Capacity Features Beat Holding the Last Measured Reference Capacity? A Reference-Aware Cross-Cell Evaluation of State-of-Health Estimation for 18650 Lithium-Ion Cells**
> Authors: Ming-Jia Cai, Hong-Jie Dai, Tsung-Hsun Wu — manuscript prepared for *IEEE Access*.
> Title from the English version v34-EN. Author order *to be confirmed by the authors*: the Chinese version v34 lists Cai, Dai, Wu; the English version v34-EN lists Dai, Cai, Wu.

Paper DOI: *to be filled in by the authors* · Zenodo DOI of this package: *to be filled in by the authors* · ORCID: *to be filled in by the authors* · License: *to be filled in by the authors* (see Section 6).

## What this package contains

From the per-row derived tables of the 34 cells in `data/`, a reader can re-run P-B/P-B+, the three reference-only baselines and the cluster bootstrap, and compare every value with the archived results. The pre-processing programs that produced those tables are included for inspection; Section 4 explains what can and cannot be re-run from raw data.

| Analysis (paper section) | Folder | Re-runnable from this package? |
|---|---|---|
| Pre-processing: raw NASA/Oxford CSV → npz → per-row features, SOH labels (Eq. 1), gates A1/A2, IC (Eqs. 3–4), feature families, cleaning — Sec. II | `code/01_download_and_convert/`, `code/02_core_modules/`, `code/07_build_derived_tables/` | Programs provided for inspection. The program converting the official downloads into the CSV files it reads is missing, the npz files are not distributed, and the raw → npz → feature path in the public setting has not been executed by the authors (Section 4). |
| Spearman screening (Eq. 6) and standardisation inside each fold; REL (Eq. 5) per cell, NASA only — Sec. II-D | `code/02_core_modules/` | Yes, as part of P-B/P-B+ |
| P-B (leave-one-cell-out) and P-B+ (sparse reference update, residual GPR, Eqs. 7–9; M = 10, 20), five regressors (Oxford without GPR-SE) — Sec. III-A, III-C, Table 4 | `code/02_core_modules/`, `code/03_run_PB_PBplus/` | Yes |
| Reference-only baselines ZOH, LIN2, LINALL on the same non-reference evaluation rows — Sec. III-B | `code/04_baselines/` | Yes |
| Cluster bootstrap (percentile and BCa, B = 10 000, seed 20260830) and the auxiliary REML random-intercept model — Sec. III-E, Supplementary Table S14 | `code/05_cluster_bootstrap/` | Yes |
| Comparison with the archived results and re-computation of the paper's numbers | `code/06_verify/` | Yes |
| Per-row derived tables of the 34 cells | `data/` | — |

**Not included** (outside the scope of this release): the P-A and P-C protocols as runnable pipelines; the configuration search and leave-one-cell-out re-selection (Table 9, Fig. 8); the 50-test Holm family and other Wilcoxon / Mann–Whitney / Hodges–Lehmann analyses; the decomposition controls (a)–(c) of Table 7; the NASA leakage variants (Table 10); the leave-one-group-out and cross-campaign splits (Table S12); the cross-period analysis with M = 5/30/50/100 and the single-reference setting (Table S7); reference noise (Table S8); artificial sparsification and measured-reference rows (Table S16); figures and manuscript build scripts.

Copied original files that still contain code this package does not use: `stage_soh_v911.py` (P-A/P-C functions), `soh_v911.py` (protocol helpers for P-A/P-C), `run_soh_v3.py` (search, P-A, P-C and the ENV/PREREG17/DATA stages; the PREREG17 stage needs a file that is not included, so do not run it, and the DATA stage is kept for reference only and has not been verified in the public setting), `config_v911.py` (search space and settings of a separate RUL study). The runner imports these files but does not call these unused parts.

---

## 1. Quick start: reproduce the main results from the derived tables

Environment of the archived run: Python 3.11.17, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, statsmodels 0.15.0, patsy 1.0.3 (`requirements.txt`), one BLAS thread per process (the runner sets this through `V3_BLAS_THREADS`, default 1). On Windows with non-ASCII paths also set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`.

statsmodels: in the archived run, statsmodels 0.15.0 and patsy 1.0.3 were installed with `pip install --no-deps`, so the formula interface ran through patsy and formulaic was not installed. To match that environment:

```bash
pip install numpy==2.4.4 pandas==3.0.2 scikit-learn==1.8.0 scipy==1.17.1
pip install --no-deps statsmodels==0.15.0 patsy==1.0.3
```

If statsmodels cannot be imported, step 3 still exits with code 0 but writes `status = 未能執行` ("could not run") for the mixed-effects model, and step 4 then reports a mismatch.

```bash
# 1) P-B and P-B+ (M = 10, 20), all models, 34 leave-one-cell-out units
python code/03_run_PB_PBplus/run_from_derived_tables.py \
       --data data/derived_per_row --qref data/cells_summary.csv --out OUT/run --workers 18

# 2) ZOH / LIN2 / LINALL on the same non-reference rows
python code/04_baselines/run_baselines.py --results OUT/run --out OUT/baselines

# 3) Cluster bootstrap + REML random-intercept model
python code/05_cluster_bootstrap/s6_cluster_bootstrap.py \
       --a04 OUT/baselines/A04_reference_only_baselines_percell.csv \
       --camp code/05_cluster_bootstrap/own_campaign_split.csv --out OUT/bootstrap

# 4) Compare with the archived per-cell results (tolerance 1e-9)
python code/06_verify/compare_to_archive.py \
       --records OUT/run/soh_records_v911.csv \
       --a04 OUT/baselines/A04_reference_only_baselines_percell.csv \
       --c7 OUT/bootstrap/c7_cluster_bootstrap.csv --s14 OUT/bootstrap/c7_S14_table_data.csv \
       --out OUT/compare_to_archive.json

# 5) Re-compute the paper's numbers (Table 5 P-B/P-B+ rows, Table 6, Table 8, Sec. IV-B bootstrap values)
python code/06_verify/reproduce_v34_numbers.py \
       --records OUT/run/soh_records_v911.csv \
       --a04 OUT/baselines/A04_reference_only_baselines_percell.csv \
       --c7 OUT/bootstrap/c7_cluster_bootstrap.csv --out OUT/reproduce_v34_numbers.json
```

The original baseline script `soh_post_v911r.py` also computes, at module level, paired Wilcoxon tests, bootstrap intervals of mean RMSE and CI-coverage summaries. These belong to analyses outside this release; `run_baselines.py` deletes those two side files (`A04_A09_B05_soh_post_summary.json`, `A12_ci95_coverage.json`) after the run, does not save the original script's standard output (it contains the out-of-scope test and coverage printouts), and records the return code, standard error and the deleted files in `log_run_baselines.txt`. Steps 3–5 use only `A04_reference_only_baselines_percell.csv`.

Results of these five steps on the original workstation (Windows 10, 20 logical cores, `--workers 18`, one BLAS thread per process, statsmodels/patsy as above), in three independent runs on 2026-10-22 (one of them by an independent reviewer):

- Step 1 produced 447 result rows (P-B 149, P-B+ M = 10 149, P-B+ M = 20 149). Runner wall-clock (`wall_sec` in `run_info.json`): 1923 s, 1938 s and 1908 s. The last run used the version 1.0 files of this package. The first two used pre-release versions of the new files (which were later revised, including the baseline wrapper's handling of side outputs); in all three runs every compared value was identical to the archived results.
- Step 4: every compared value was identical to the archived results (maximum absolute difference 0 at tolerance 1e-9). Step 5: all 145 re-computed numbers matched the paper. `reproduce_v34_numbers.py` rounds with Python's format, which rounds half to even; this matters for one value, the NASA mean n_ref at M = 10 (16.25, printed 16.2 in Table 8).

**Cross-platform caveat (paper Sec. III-F).** On a different platform the features and the Ridge and SVR errors were reproduced, but the GPR and MLP errors were not bit-identical. Expect small differences in GPR/MLP-based numbers outside the original environment; `compare_to_archive.py` reports every difference rather than hiding it.

Reading the tables: the runner parses the CSV files with `float_precision="round_trip"`. With pandas' default parser the features differ from the in-memory values in the last digits (up to about 2e-13); in a test run this changed MLP RMSEs by up to about 1.3 pp and SVR RMSEs by up to about 0.04 pp.

---

## 2. Interpretation limits (from the paper)

- All comparisons of P-B+ with the baselines and the label strata are post hoc and exploratory (Sec. I).
- The cluster bootstrap and the mixed-effects model are, in addition, not adjusted for multiple comparisons (Sec. III-E). The Holm-adjusted tests reported in the paper are not part of this package.
- Cluster-bootstrap intervals for subsets with no more than 4 clusters are unreliable; with one cell per NASA cluster the mixed-effects model is not identifiable (Sec. III-E, Table S14).
- The pipeline is offline and not causal as a whole: gate A2 uses labels after the step, feature cleaning uses a centred window, REL divides each cell's features (including the held-out test cell's) by the median of its own first three retained rows, and Q_ref is the maximum over the whole record (Sec. II-B, II-C, II-D, III-A). Only the residual correction of P-B+ is causal.
- The locked configuration was chosen on the same cells (Sec. III-E), so its numbers are optimistic for the original 27 cells; P-B+ uses the test cell's true, noise-free laboratory capacities at the reference rows.
- All results use a single seed (20260830; residual-GPR restarts seed 0). GPR hyper-parameters, log marginal likelihoods and convergence status were not stored, and optimiser convergence warnings are suppressed (Sec. III-C, III-F).

## 3. Derived data (`data/`)

- `data/derived_per_row/<source>__<cell>.csv` — one file per cell (4 NASA, 9 own NCR18650GA, 21 Oxford; 16 567 cycle-index rows in total). Columns on the left are the archived per-row frame exactly as used by the locked-configuration run: SOH label, capacity, temperature, the candidate IC / ΔQ features of the source's selected window and filter, cycle axes, and gate flags. Columns on the right, prefixed `ann_`, are annotations added for this release: measured vs. interpolated label, A1 pass, A2 drop, in-model flag, evaluation index *i*, residual-GPR cycle axis *c_i*, and reference-row flags for M = 10 and M = 20 (*i* mod M = 0).
- `data/cells_summary.csv` — one row per cell: cluster used by the bootstrap (Oxford ageing group, own test campaign, NASA cell), label stratum (measured 25 / interpolated 9), Q_ref, row counts, rows removed by A1/A2, measured-reference fraction, number of reference rows and evaluation rows for M = 10 and 20.
- `data/DATA_DICTIONARY.md` — definition, unit and code origin of every column.

The features are the full candidate set; the files contain no screening, standardisation or REL. Inside each fold the code standardises with training rows only and screens features with training cells only. Screening as implemented (`soh_v911.select_features`): Spearman ρ is computed per training cell; a feature is kept if the minimum |ρ| over the training cells is at least 0.85; kept features are ranked by mean |ρ| and the top 12 are used. If fewer than 12 pass, the code falls back to ranking all candidates; in the locked-configuration run this fallback was never triggered (checked for all 34 folds). The number of candidates passing the threshold per fold was 21 for NASA, 18–19 for the own cells and 12–13 for Oxford, so for 20 of the 21 Oxford folds exactly 12 features passed and all of them were used. REL (NASA only) is applied per cell, test cell included, as stated in Sec. II-D. The runner drops all `ann_` columns before modelling.

---

## 4. From raw data (NASA and Oxford only)

`code/01_download_and_convert/` holds the scripts that turned the NASA and Oxford raw CSV files into the npz datasets, including the removal of the Oxford Part 1/Part 2 join artefacts and row-level duplicates (Sec. II-B). Read `download_instructions.md` and `README_convert.md` first; `check_raw_files.py` checks (offline) that local raw files have the expected names, headers, row counts and, for the nine deduplicated Oxford files, SHA-256 digests. The npz → per-row feature step is the DATA stage of the original driver (`README_convert.md`, Section 2C); `code/07_build_derived_tables/` records how the archived frames were turned into `data/`.

Known gaps, stated plainly:

- The program that converted the official NASA and Oxford downloads into the 11-column CSV files read by `extract_nasa2.py` / `extract_ox2.py` is not in the authors' archive. Users must perform that conversion themselves and can check the result against the row counts (and SHA-256 for the nine Oxford files) listed by `check_raw_files.py`.
- As stated in the paper, the original Oxford Research Archive Part 1/Part 2 files were not obtained, so no file-by-file comparison with them was made.
- As stated in the paper (Sec. III-F), rebuilding from the raw files reproduced the labels of the original 27 cells to within 1.4 × 10⁻¹⁴, but the Q(V) grid changed slightly for reasons that could not be determined. A rebuild on another machine may therefore not be bit-identical.
- The path that uses only NASA and Oxford raw data has not been executed by the authors in the public setting (`README_convert.md`, Section 5).
- Several copied scripts in `code/01_download_and_convert/` contain hard-coded paths of the authors' machine; they are listed by line in `README_convert.md`. Two copied files elsewhere also contain such paths: `code/02_core_modules/pv911_run_SOH/02_programs/data_v911.py` L11 (in a comment) and `code/04_baselines/common_v911r.py` L5–6 (a default results path, which `run_baselines.py` overrides through the environment variable `PV911_RESULTS`). The original files were not edited.
- The authors' own NCR18650GA raw time series are not public (paper, "Data availability"), and the scripts that convert them are not included. The nine own cells can be re-analysed only from the derived tables.

---

## 5. Code layout and provenance

```
code/
  01_download_and_convert/   NASA/Oxford raw CSV -> npz; Oxford de-duplication (original scripts) + new download notes / file checker
  02_core_modules/           original analysis modules (IC, features, gates, models, P-B/P-B+ protocols) and the original parallel driver
  03_run_PB_PBplus/          NEW runner that starts from data/derived_per_row
  04_baselines/              original baseline script + NEW wrapper
  05_cluster_bootstrap/      original bootstrap / REML script + own-campaign map
  06_verify/                 NEW comparison and number re-computation scripts; expected/ = archived per-cell results
  07_build_derived_tables/   NEW script that built data/ from the archived frames, with its run log
data/                        derived per-row tables, per-cell summary, data dictionary
```

- Files copied from the authors' archive are byte-identical to the originals. `MANIFEST.csv` lists, for every file except itself and `SHA256SUMS.txt`, its origin path in the archive (or `NEW`), its SHA-256, its role and the paper section it serves. `SHA256SUMS.txt` covers the whole package.
- Copied files keep the authors' original comments and folder names, which refer to internal analysis-plan versions, a separate RUL study and the manuscript revision process. These comments are not claims of the paper.
- Every new file states in its header which original code it calls or copies, with line numbers.
- Fixed settings: locked configuration of analysis-plan version 17 (Table 4), global seed 20260830, residual-GPR restart seed 0, bootstrap B = 10 000 with seed 20260830. These are taken from the original code and are not exposed as options.

---

## 6. Data sources and licences

- NASA battery data set (B. Saha and K. Goebel, NASA Ames Prognostics Center of Excellence, 2007; cells B0005, B0006, B0007, B0018): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ — licence of the NASA data: *to be filled in by the authors*.
- Oxford path-dependent degradation data (T. Raj, University of Oxford; Part 1 2020, Parts 2 and 3 2021), Oxford Research Archive, Open Data Commons Open Database License: Part 1 https://doi.org/10.5287/bodleian:v0ervBv6p · Part 2 https://doi.org/10.5287/bodleian:2zvyknyRg · Part 3 https://doi.org/10.5287/bodleian:j1a2eD7ow. The Oxford rows of `data/` are derived from these ODbL data.
- Own NCR18650GA cells (cell2, cell3, GA1–GA7): raw data not public; only the derived per-row tables are included.
- Licence of this code and of the derived tables: *to be filled in by the authors*.

## 7. Citation

*To be filled in by the authors after acceptance (paper DOI and Zenodo DOI).*
