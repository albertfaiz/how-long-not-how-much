# How Long, Not How Much

Code, result files and figures for:

> Ahmad, F.; Lary, D.J.; Ruwali, S.; Shrestha, S.; Aker, A.; Waczak, J.; Niloy, M.S.H.; Patra, R.; Awoyinka, T.D. Wijeratne, L.O.H.; Iqbal, M.
> *How Long, Not How Much: Summarising Satellite and Gridded Data Reorders Machine Learning Predictors of U.S. County Life Expectancy.* Manuscript for *Sensors* (MDPI), 2026.

A gradient-boosted model (XGBoost) predicts county life expectancy in the contiguous United States from satellite and gridded environmental data only. The panel covers 3,077 counties from 2003 to 2019, or 52,309 county-years. Tested on counties it never saw, the model reaches an out-of-fold R² of 0.666.

The paper holds the data fixed and changes only the summary statistic of a field, then asks four questions:

- How far does the change move the model's attribution (SHAP)?
- Does atmospheric chemistry explain which description the model uses?
- Which turning points of the attribution curves are stable under noise and a split of the country?
- Where does a model of the physical environment err?

Every number, table and figure of the paper is produced by the code here, from the result files here. The two large inputs are archived at Zenodo.

## Data (Zenodo)

Data DOI: https://doi.org/10.5281/zenodo.23353493

| File | Put it at | What it is |
|---|---|---|
| `Final_ML_Ready_Dataset.csv` (unzip `Final_ML_Ready_Dataset.csv.zip`) | repository root | The county-year table: 74,863 rows by 1,318 columns, 2000 to 2023. It is the input of the model run. |
| `studyA_reference_frame.parquet` | `results_studyA_20260801_004313/` | The same table with the 46 engineered columns added: 53,107 rows, 2003 to 2019. The analysis steps that refit the model read it. |

The SHA-256 checksums are in the Zenodo record (`SHA256SUMS.txt`).

## Layout

```
pipeline_core_v2.py, engineered_features_v2.py,
pipeline_A_moisture_v2.py, run_studies_v2.sh, paths.py
                                 the model run (environment only, 2003 to 2019)
superset_311981.log              the log of that run (UT Dallas Juno cluster, 1 August 2026)
results_studyA_20260801_004313/  its outputs: the fitted model, out-of-fold predictions,
                                 the SHAP ranking, the admission record and summary tables
data_collection/                 the Google Earth Engine and merge scripts that built
                                 the county-year table (provenance)
paper_final/
  main.tex, references.bib       the manuscript (MDPI Sensors LaTeX)
  constants_*.tex, tables_*.tex  every number and table the manuscript prints (generated)
  Figures/                       Figures 1 to 8 and the graphical abstract (600 dpi PNG and PDF)
  analysis_final/                the analysis of the paper (see its README)
    paper4_analysis_all_in_one.py   all analysis steps and the constants builder, in one file
    paper4_figures_all_in_one.py    all figures, in one file
  derived/                       the result files of the analysis steps (JSON and CSV),
                                 with one manifest per step
    fold_maps/                   the cross-validation fold of every row (see Software)
  pipeline/                      a copy of the model-run code, which step 07 reads
```

## Checking the paper's numbers (minutes)

The commands below use only the result files in this repository:

```
conda env create -f environment.yml
conda activate satsdoh
cd paper_final/analysis_final
python3 paper4_analysis_all_in_one.py --verify            # stored code = the files in this folder
python3 paper4_analysis_all_in_one.py --run constants     # rewrites every number and table
python3 paper4_figures_all_in_one.py --check-only         # draws all figures and checks the layout
python3 paper4_figures_all_in_one.py --outdir ../Figures  # writes the figures
```

`--run constants` rewrites `paper_final/constants_*.tex` and the three `tables_*.tex` files. It also checks that every sentence describing a number in words (for example "weakly clustered") still matches the number.

## Rerunning the analysis (hours)

With the two Zenodo files in place:

```
bash run_all.sh
```

This runs every analysis step in order, then the constants builder and the figures. The steps that refit the model or draw bootstrap replicates take minutes to hours each.

The model run itself used the Juno cluster with a GPU:

```
bash run_studies_v2.sh
```

Its settings are fixed in `pipeline_core_v2.build_models` (XGBoost with 1,200 trees, learning rate 0.03 and maximum depth 8). They match the saved model, as step 07 checks.

## Compiling the manuscript

`paper_final/main.tex` uses the MDPI LaTeX template. Download the template from MDPI and place its `Definitions/` folder beside `main.tex`, or upload the files to an MDPI Overleaf project. Then compile with pdflatex, bibtex, pdflatex and pdflatex.

## Software

The analysis steps ran in Python 3.11.13 with the versions in `environment.yml`. The model run used Python 3.10.12 on Juno. Every script is seeded.

Use the pinned versions to reproduce the numbers to the last digit. Two points make the code safe under other versions as well:

- **Folds.** scikit-learn's `GroupKFold` does not assign groups of equal size to the same folds in every version, and every county here has 17 rows. The scripts therefore read the fold of every row from `paper_final/derived/fold_maps/` and do not ask scikit-learn for it again.
- **Saved models.** XGBoost 3.0 cannot load a saved regressor under scikit-learn 1.8 or later. The scripts then read the trees through the booster, which gives the same predictions.

Every script prints a note when an installed library differs from the pinned version.

## Citation and licence

See `CITATION.cff`. Code: MIT licence (`LICENSE`).
