# How Long, Not How Much

Code, result files and figures for:

> Ahmad, F.; Lary, D.J.; Ruwali, S.; Shrestha, S.; Aker, A.; Waczak, J.; Niloy, M.S.H.; Patra, R.; Awoyinka, T.D.; Wijeratne, L.O.H.; Iqbal, M.
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
| `Final_ML_Ready_Dataset.csv.zip` (unzip `Final_ML_Ready_Dataset.csv`) | repository root | The county-year table: 74,863 rows by 1,318 columns, 2000 to 2023. It is the input of the model run. |
| `data_dictionary_engineered_columns.csv` | repository root | Metadata dictionary containing definitions and formulas for engineered columns. |

## Layout

```
pipeline_core_v2.py, engineered_features_v2.py,
pipeline_A_moisture_v2.py, run_studies_v2.sh, paths.py
                        the model run (environment only, 2003 to 2019)
superset_311981.log     the log of that run (UT Dallas Juno cluster, 1 August 2026)
results_studyA_20260801_004313/  its outputs: the fitted model, out-of-fold predictions,
                        the SHAP ranking, the admission record and summary tables
data_collection/        the Google Earth Engine and merge scripts that built
                        the county-year table (provenance)
paper_final/
  main.tex, references.bib    the manuscript (MDPI Sensors LaTeX)
  constants_*.tex, tables_*.tex  every number and table the manuscript prints (generated)
  Figures/              Figures 1 to 8 and the graphical abstract (600 dpi PNG and PDF)
analysis_final/         the analysis of the paper (see its README)
  paper4_analysis_all_in_one.py  all analysis steps and the constants builder, in one file
  paper4_figures_all_in_one.py   all figures, in one file
derived/                the result files of the analysis steps (JSON and CSV),
                        with one manifest per step
fold_maps/              the cross-validation fold of every row (see Software)
pipeline/               a copy of the model-run code, which step 07 reads
```

## Rerunning the analysis

To rerun the analysis steps from scratch (which regenerates every JSON, CSV, figure, and LaTeX constant file):

```bash
cd analysis_final
python3 paper4_analysis_all_in_one.py
python3 paper4_figures_all_in_one.py
```

With the two Zenodo files in place:

```bash
# Optional: refit the model on the full 53k dataset (requires ~16 GB RAM)
./run_studies_v2.sh
```

## Software and environment

The analysis was executed on Linux (x86_64) using Python 3.10 with:
- `xgboost` 2.0+
- `shap` 0.44+
- `scikit-learn` 1.3+
- `pandas` 2.1+, `numpy` 1.26+, `scipy` 1.11+
- `matplotlib` 3.8+, `seaborn` 0.13+

The cross-validation assignment for every row is deterministic and recorded in `fold_maps/`.

## License

Code is licensed under the MIT License. Data and derived result files are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
