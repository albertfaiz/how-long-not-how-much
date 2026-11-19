# analysis_final: the analysis of the paper

Two self-contained files reproduce everything the manuscript prints.

| File | What it holds | How to use it |
|---|---|---|
| `paper4_analysis_all_in_one.py` | Stored verbatim: the 25 analysis steps (24 that produce results the paper prints, and one that checks its method statements), the constants builder (every number and Tables 1 to 4), `_config.py` and the model-run pipeline | `--list`, `--run 02`, `--run constants`, `--show 25`, `--verify` |
| `paper4_figures_all_in_one.py` | Figures 1 to 8 and the graphical abstract, 600 dpi PNG and PDF, with a layout check | `--only 3,5`, `--check-only`, `--outdir ../Figures` |

The same code is also here as separate scripts, numbered in run order. Steps marked "refits" in `--list` fit XGBoost again or draw bootstrap replicates and take minutes to hours. Steps marked "reads" work from the result files in `../derived/` and take seconds.

## What each step gives the paper

| Step | Paper |
|---|---|
| 01 | checks the environment and inputs |
| 02 | error by outcome bracket (Figure 2C) |
| 03 | five baselines (Figure 2A) |
| 04 | drop-one refit of night cooling efficiency (Figure 4, Section 3.2) |
| 05, 06 | attribution sample and bootstrap turning points (Figure 6, Table 4) |
| 07 | learner settings and skill (Table 2) |
| 10 | reruns a step and compares its outputs |
| 11 | the two night-cooling constructions identified by search (Section 2.4) |
| 12 | fields with several summaries and their attribution spread (Section 3.2) |
| 13 | state hold-out and residual clustering, with Moran's I of the residual and of the outcome (Sections 3.1, 3.6) |
| 14 | regional invariance of turning points (Figure 7C) |
| 16 | frozen modelling panels and the coverage-gate audit |
| 17 | refits without the formaldehyde durations (Section 3.2) |
| 19 | temporal validation (Section 3.1) |
| 20 | null skill and null turning points (Figures 2D, 7B) |
| 22 | the all-columns reference refit (Section 2.7) |
| 23 | influence and predictor noise (Figure 7D, Section 3.6) |
| 25 | duration advantage against atmospheric lifetime (Figure 5C, D; Table 3) |
| 26 | learning curve and rank stability (Figure 2B) |
| 28, 30 | residual audit against socioeconomic data and its sensitivity (Figure 8) |
| 33 | the data behind Figure 8C |
| 34 | library versions (Section 2.6) |
| 35 | checks the sentences that say how something was done, or what the data show without a number, against the code and the result files (24 checks) |
| constants | every number of the text and Tables 1 to 4 |

Some manifests in `../derived/manifests/` name the scripts that originally wrote a file under their earlier names:

| Earlier name | Name here | What the earlier script also did |
|---|---|---|
| `12_summary_statistic_census_and_matched_refit.py` | step 12 | a census test and a matched refit |
| `14_information_theory_and_regional_invariance.py` | step 14 | a mutual-information comparison |
| `17_formaldehyde_pair_anatomy.py` | step 17 | an anatomy of the two formaldehyde durations |
| `20_null_calibration_and_multiplicity.py` | step 20 | a census null and a multiplicity family |

Those extra analyses are not part of this paper. The functions kept here are unchanged from the earlier scripts.

No synthetic or mock data is used anywhere: every step reads the project's own files.

## Folds

Every step takes its grouped folds from `_config.grouped_splits`, which reads the fold of every row from `../derived/fold_maps/`. scikit-learn's `GroupKFold` does not give the same partition in every version when all groups have the same size, as they do here (17 rows per county), so the partition is stored and not recomputed. Baselines and refits therefore share one partition, and a stored fold prediction always meets the rows it was made for.

## Rebuilding the two all-in-one files

```
python3 _builders/build_analysis_final.py .. paper4_analysis_all_in_one.py _builders/analysis_template_final.py
python3 _builders/build_figs.py . _builders/template.py paper4_figures_all_in_one.py
```
