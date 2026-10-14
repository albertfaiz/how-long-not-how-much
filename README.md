# Predictive Multiplicity in Environmental Health

Code, result files and figures for:

> Ahmad, F.; Lary, D.J.; Ruwali, S.; Shrestha, S.; Aker, A.; Waczak, J.; Niloy, M.S.H.; Patra, R.; Awoyinka, T.D.
> *Predictive Multiplicity in Environmental Health: Information Theory Exposes How Summary Statistics Shape Machine Learning Attribution.* Submitted to *Sensors* (MDPI), 2026.

A gradient-boosted model predicts county life expectancy in the contiguous United States (52,309 county-years, 3,077 counties, 2003 to 2019) from satellite and gridded environmental data only. The paper asks how much of the model's attribution ranking reflects the environment and how much reflects the summary statistic chosen for each field. It then asks what survives: atmospheric lifetime, stable turning points and the geography of the model's error.

Every number, table and figure in the paper is produced by the scripts here, from the result files here, except for two large inputs archived at Zenodo.

## Data (Zenodo)

Data DOI: https://doi.org/10.5281/zenodo.23353493

Download and place:

| File | Put it at | What it is |
|---|---|---|
| `Final_ML_Ready_Dataset.csv` (unzip `Final_ML_Ready_Dataset.csv.zip`) | repository root | county-year table, 74,863 rows x 1,318 columns, 2000 to 2023; the input of the model run |
| `studyA_reference_frame.parquet` | `results_studyA_20260801_004313/` | the same table after the 46 engineered columns are added (53,107 rows, 2003 to 2019); the analysis scripts read this |
| `panel_all_admitted.csv`, `panel_satellite_only.csv` (optional) | `paper_final/derived/` | the exact matrices the model was fitted on (581 and 239 columns) |

The SHA-256 checksums are in the Zenodo record (`SHA256SUMS.txt`).

## Layout

```
data_collection/                     Google Earth Engine and merge scripts that built the county-year table
pipeline_core_v2.py, engineered_features_v2.py,
pipeline_A_moisture_v2.py, run_studies_v2.sh   the model run (study A: environment only, 2003 to 2019)
superset_311981.log                  log of that run (UT Dallas Juno cluster, 1 August 2026)
results_studyA_20260801_004313/      its outputs: fitted model, out-of-fold predictions, SHAP ranking,
                                     admission record, tables and figures (the reference frame is on Zenodo)
paper_final/analysis/                the paper's analysis battery, scripts 01 to 34, and the figure and constants builders
paper_final/derived/                 every result file the battery wrote (JSON and CSV), with a manifest per script
paper_final/Figures/                 Figures 1 to 10 and the graphical abstract (PNG and PDF)
paper_final/tables_v3/, *.tex        generated LaTeX tables and the constants the manuscript reads
```

`run_ml_satallite_and_SDOH.py` and `pipeline_B_vanishing_v2.py` are included for provenance (the earlier nocturnal study and the socioeconomic study). This paper does not use their outputs.

## Reproducing the paper

Environment: Python 3.11 with the versions in `environment.yml` (the paper's numbers were produced with exactly these).

```
conda env create -f environment.yml
conda activate satsdoh
bash run_all.sh            # scripts 01 to 34, then constants and figures
```

`run_all.sh` runs the analysis battery in numerical order, then `build_paper_constants.py` (writes `paper_final/constants_*.tex` and `tables_generated.tex`) and the figure scripts. Several scripts refit the model and take minutes to hours on a laptop; each keeps checkpoints and resumes. `build_paper_constants.py` ends by checking every sentence of the paper that states a quality of a number ("not significant", "barely move") against the new values and prints WARN if one no longer holds.

To refit the model itself (optional; the original run used one GPU node of the Juno cluster):

```
python3 pipeline_A_moisture_v2.py --csv Final_ML_Ready_Dataset.csv \
    --outdir results_studyA_<stamp> --folds 8 --coverage 0.95 --n-boot 1000 --forward --gpu
```

### Determinism and library versions

Every script is seeded. Script 01 shows that the fitted model is identical at 1 and 4 threads, and script 10 reruns a script and compares its outputs by hash. Across library releases, results that refit XGBoost can move slightly. Under XGBoost 3.2.0 instead of 3.0.2, four results moved: the state hold-out R2 (0.504 against 0.509), Moran's I of the residual (0.035 against 0.032), the perturbation shifts of the turning points (second decimal place) and the permutation counts (up to 6 of 47 fields in a split). No conclusion changed. Script 34 records this, and the earlier run's files are in `paper_final/derived/reference_run_2026-10-05/`.

### Data versions

Script 32 checks the data files against each other cell by cell. `Final_ML_Ready_Dataset.csv` equals the reference frame in all 1,313 shared columns. The input of the earlier nocturnal study (`Master_Cleaned_NoArtifacts_With_LIVE_Prefix.csv`, not needed here) equals it in all 1,160 shared columns; the files differ only in which columns and years each holds.

### Figure 10C

`paper_final/derived/figure10c_residual_and_indicators.csv` (script 33) holds, for each of 3,077 counties, the mean out-of-fold residual and the six socioeconomic indicators named in advance. The indicators are county means over the years each is observed, mostly 2009 to 2019. None entered the model. The file reproduces every bar of the panel.

## Earth Engine

The data-collection scripts read the Earth Engine project from the `EE_PROJECT` environment variable. They were written as Colab notebooks and keep their original Drive paths.

## Licence and citation

Code: MIT (see `LICENSE`). Data on Zenodo: CC BY 4.0. Please cite the paper and the Zenodo record (see `CITATION.cff`).

## Related work this builds on

- Ahmad, F. et al. Nighttime Thermal Patterns and County Life Expectancy. *Remote Sensing* 2026, 18, 2330. https://doi.org/10.3390/rs18142330 (code: https://albertfaiz.github.io/Multimodal_geo_fusion_FM/)
- Shrestha, S. et al. The External Exposome and Life Expectancy. *Air* 2026, 4, 10. https://doi.org/10.3390/air4020010
- Shrestha, S. et al. Atmospheric Exposures and Cardiovascular Mortality in United States Counties. *AI Sensors* 2026, 2, 8. https://doi.org/10.3390/aisens2030008
- Ruwali, S. et al. Multi-Modal Sensing for Estimating U.S. Life Expectancy. *Preprints* 2026, 2026041997. https://doi.org/10.20944/preprints202604.1997.v1
