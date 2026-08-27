# Reproducing the analysis

Everything reported in the paper is produced by the code in this repository.
This document walks through it end to end.

## 1. Environment

The pipeline was developed and run with Python 3.11 on the
UT Dallas high-performance computing cluster, using NVIDIA H200 accelerators. It runs on CPU alone; the `--gpu`
flag only shortens the learner tournament.

```bash
conda env create -f environment.yml
conda activate how-long-not-how-much
```

If you prefer pip, use `requirements.txt`. Note that `geopandas` and `rasterio`
install more reliably through conda. If pip gives you trouble on those two, use
the conda route.

Verify the environment:

```bash
python -c "import numpy, pandas, sklearn, shap, xgboost, catboost; print('ok')"
```

## 2. Data

The assembled county-year panel is deposited on Zenodo at
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).
Download it and place it under `data/`:

```
data/
├── Final_ML_Ready_Dataset.csv      the assembled county-year panel
├── studyA_reference_frame.parquet     the exact frame the model used
└── studyA_final_model.joblib          the fitted model
```

See [`../data/README.md`](../data/README.md) for file sizes and checksums.

You do not need to re-run the pipeline to check a number in the paper. Every
result table is already in `tables/`, and the reference frame lets you verify
any intermediate quantity directly.

## 3. Run the pipeline

```bash
python code/pipeline_A_moisture_v2.py \
    --csv data/Final_ML_Ready_Dataset.csv \
    --outdir results_studyA \
    --folds 8 \
    --gpu
```

Expected runtime: roughly two to five hours on a single GPU node, dominated by
the learner tournament and the interventional attribution. Add `--fast` for a
smoke test with fewer trees, which finishes in minutes and is useful for
verifying the environment before committing the full run.

The run writes, into `results_studyA/`:

| File | Contents |
|---|---|
| `*_feature_gate.csv` | every admission and exclusion with its reason |
| `*_model_tournament.csv` | learner comparison under the identical protocol |
| `*_oof_predictions.csv` | out-of-fold predictions and residuals |
| `*_cv_folds.csv` | per-fold metrics |
| `*_shap_ranking.csv` | interventional attribution ranking |
| `*_final_model.joblib` | fitted model, feature list, imputation values |
| `*_metrics.json` | every headline number in one machine-readable file |

## 4. Rebuild the tables and figures

```bash
python code/studyA_analysis.py          # writes tables/
python code/studyA_maps.py  # writes figures/
```

The analysis script prints a `PASTE-BACK SUMMARY` block containing every number
the manuscript needs. Those values populate the LaTeX constants file in
`paper/`, which is why the text and the results cannot disagree.

## 5. Verify against our run

Compare your `tables/*_summary.json` against ours, committed in this
repository. The headline values you should reproduce:

| Quantity | Expected |
|---|---|
| Out-of-fold R-squared | 0.666 |
| Winning learner | XGBoost |
| County-years | 52,309 |
| Attributed features | 581 |
| Modalities | 18 |

Random seeds are fixed throughout, so a rerun on the same data reproduces the
same numbers. Small differences in the last decimal place can arise from
threading nondeterminism in the boosted-tree libraries; anything larger
indicates a genuine discrepancy and we would like to hear about it.

## 6. If something does not match

Please [open an issue](https://github.com/albertfaiz/how-long-not-how-much/issues) with your Python version,
package versions (`conda list`), and the differing values. Reproducibility
reports are welcome and we will investigate them.
