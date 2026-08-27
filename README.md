<div align="center">

# ⏳ How Long Not How Much

### Exposure duration outranks concentration in predicting United States county life expectancy

[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX) [![License](https://img.shields.io/badge/code-MIT-green)](LICENSE) [![Data](https://img.shields.io/badge/data-CC-BY-4.0-lightgrey)](data/README.md) [![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](environment.yml) [![Pages](https://img.shields.io/badge/website-live-brightgreen)](https://albertfaiz.github.io/how-long-not-how-much/)

**[Read the paper](#citation) &nbsp;|&nbsp; [Project website](https://albertfaiz.github.io/how-long-not-how-much/) &nbsp;|&nbsp; [Data on Zenodo](https://doi.org/10.5281/zenodo.XXXXXXX) &nbsp;|&nbsp; [Reproduce]( docs/REPRODUCE.md)**

</div>

---

## Overview

This repository contains the complete analysis code, figures, and result tables
for:

> **How Long, Not How Much: Exposure Duration Outranks Concentration in an Eighteen-Modality Satellite and Reanalysis Fusion for United States County Life Expectancy**
> Faiz Ahmad, David J. Lary, Shisir Ruwali, Samyak Shrestha, Adam Aker, John Waczak, Prabuddha Madushanka
> *Sensors (MDPI)*, 2026.

The study predicts county mean life expectancy at birth across the contiguous
United States from environmental sensing data, using a leakage-controlled
machine-learning pipeline and interventional attribution. Everything needed to
reproduce every number in the paper is here.

| Metric | Value |
|---|---|
| R² | **0.666** |
| MAE | **1.03 yr** |
| county-years | **52,309** |
| modalities | **18** |
| features | **581** |
| within 2 yr | **87.6%** |

---

## Key findings

### 1. The leading predictor is a duration, not a concentration

The fraction of the year formaldehyde spends above its 75th percentile ranks #1 of 581 features, with a mean absolute attribution of 0.341 years of life expectancy.

### 2. The same molecule, measured two ways, differs 71-fold

The annual mean column amount of formaldehyde ranks #196. The difference is the summary statistic, not the species.

### 3. The advantage follows atmospheric lifetime

Duration wins for 7 of 8 short-lived species but only 3 of 12 longer-lived ones (Mann-Whitney p = 0.006, Spearman rho = -0.47). A numerical artefact would not sort by chemistry.

### 4. Coverage auditing changes the conclusions available

238 features were removed because their record does not span the study window. Without that gate their importance would have rested on imputed values.

### 5. Model error tracks social divergence

Absolute error rose 27.7% across the record while county ordering held, and annual error tracked the life-expectancy Gini coefficient at r = 0.989.

---

## What makes this pipeline different

Three design choices distinguish this work from a conventional multi-sensor
health model, and each is implemented in code you can inspect.

**A temporal coverage gate.** A feature bank assembled from many sensors mixes
records of very different length. A variable present for only part of the study
window must be imputed for the rest, and any importance it then receives is
partly an artefact of the fill. Every feature must be observed for a stated
fraction of the study window before it is admitted, and every admission and
exclusion is written to an auditable report.

**A leakage blocklist.** The candidate bank contains columns that are themselves
functions of mortality. Including any of them would let the model predict life
expectancy from life expectancy. These are blocked before modelling, by an
explicit list and a pattern match, and the block count is printed in every run
log.

**County-grouped cross-validation with fold-internal preprocessing.** County
life expectancy varies far more between counties than within a county over two
decades, so a random split would let the model recognise a county rather than
learn a transferable relationship. All years of any county fall in a single
fold, and median imputation is fitted on the training fold only. There is no
global imputation anywhere in the pipeline.

---

## Repository layout

```
how-long-not-how-much/
├── code/                     analysis pipeline (see table below)
├── figures/                  publication figures, PNG at 300 dpi and vector PDF
├── tables/                   every result table cited in the paper, as CSV
├── data/                     pointer to the Zenodo deposit, see data/README.md
├── docs/
│   └── REPRODUCE.md          step-by-step reproduction instructions
├── paper/                    LaTeX sources, constants block, bibliography
├── environment.yml           conda environment
├── requirements.txt          pip alternative
├── CITATION.cff              machine-readable citation
└── index.html                project website source
```

### Code

| File | Purpose |
|---|---|
| `engineered_features_v2.py` | Physically motivated engineered layer, all formulas documented |
| `pipeline_core_v2.py` | Coverage gate, leakage blocklist, grouped CV, learner tournament, interventional SHAP |
| `pipeline_A_moisture_v2.py` | Study A driver: environment-only model, 2003-2019 |
| `studyA_analysis.py` | All statistics reported in the paper |
| `studyA_extra_analysis.py` | Lifetime-split and cross-validation stability figures |
| `studyA_plots.py` | Figures 1 to 9 |
| `studyA_maps.py` | Maps 1 to 4 |
| `paper_style_v2.py` | Shared palette and publication feature naming |

### Figures

| Figure | Shows |
|---|---|
| `figures/A_fig1_global_ranking.png` | Global predictor ranking coloured by physical group |
| `figures/A_fig2_duration_vs_concentration.png` | Duration against concentration, aggregate and per molecule |
| `figures/A_fig10_lifetime_split.png` | The duration advantage against atmospheric lifetime |
| `figures/A_fig3_group_hierarchy.png` | Attribution by physical group, total and per feature |
| `figures/A_fig4_thermal_index_audit.png` | Pure against composite thermal indices |
| `figures/A_fig5_dewpoint_threshold.png` | Moisture profile and the temperature-stratified threshold |
| `figures/A_fig6_orthogonality.png` | Collinear inputs, independent attributions |
| `figures/A_fig7_dose_response.png` | Learned dose-response curves |
| `figures/A_fig8_stability_error.png` | Stability, error structure, and inequality |
| `figures/A_fig9_modality_gate.png` | Modality contribution and the coverage gate |
| `figures/A_fig11_cv_stability.png` | Cross-validation and learner-tournament stability |
| `figures/A_Map1_observed_vs_predicted.png` | Observed and predicted life expectancy |
| `figures/A_Map2_residual_geography.png` | Residual geography |
| `figures/A_Map3_leading_exposure.png` | Geography of the leading exposure |
| `figures/A_Map4_longitudinal_shift.png` | Observed against predicted change |

---

## Quick start

```bash
git clone https://github.com/albertfaiz/how-long-not-how-much.git
cd how-long-not-how-much

# option 1: conda (recommended, pins geospatial dependencies cleanly)
conda env create -f environment.yml
conda activate how-long-not-how-much

# option 2: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download the input data from Zenodo, see data/README.md
python code/pipeline_A_moisture_v2.py --csv data/Final_ML_Ready_Dataset.csv --gpu

# then regenerate every table and figure
python code/studyA_analysis.py
python code/studyA_maps.py
```

Full instructions, including expected runtimes and how to verify your output
against ours, are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Data sources

Input data are deposited on Zenodo under
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX). The
deposit contains the assembled county-year panel, the fitted model, and every
intermediate result table, so a reviewer can verify any number in the paper
without re-running the pipeline. The primary sources are listed below and are
all publicly available.

| Source | Role | Link |
|---|---|---|
| IHME county life expectancy | Outcome variable, county mean life expectancy at birth | [link](https://ghdx.healthdata.org/) |
| CAMS global reanalysis | Atmospheric composition: reactive gases and speciated aerosol | [link](https://ads.atmosphere.copernicus.eu) |
| ERA5 reanalysis | Atmospheric state: temperature, humidity, wind, pressure | [link](https://cds.climate.copernicus.eu) |
| PRISM climate | Interpolated US surface temperature, dew point, precipitation | [link](https://prism.oregonstate.edu) |
| TerraClimate | Monthly water balance and surface climate | [link](https://www.climatologylab.org/terraclimate.html) |
| MODIS via Google Earth Engine | Land-surface temperature and vegetation indices | [link](https://earthengine.google.com) |
| Landsat via Google Earth Engine | Surface reflectance and derived spectral indices | [link](https://earthengine.google.com) |
| Global Human Modification | Cumulative landscape transformation index | [link](https://doi.org/10.1111/gcb.14549) |
| SoilGrids 2.0 | Soil bulk density, texture, pH, organic carbon | [link](https://soilgrids.org) |
| Harmonised night-time lights | DMSP and VIIRS reconciled long series | [link](https://doi.org/10.1038/s41597-020-0510-y) |
| County Health Rankings | Social determinants of health (Study B only) | [link](https://www.countyhealthrankings.org) |
| American Community Survey | Demographics, income, education, housing (Study B only) | [link](https://www.census.gov/programs-surveys/acs) |

---

## Computational environment

Analyses were run on the UT Dallas high-performance computing cluster using NVIDIA H200 accelerators. The pipeline
runs on CPU alone; the GPU flag only shortens the learner tournament.

- Python 3.11
- Key packages:

```yaml
  - numpy>=1.26
  - pandas>=2.1
  - scikit-learn>=1.4
  - xgboost>=2.0
  - lightgbm>=4.3
  - catboost>=1.2
  - shap>=0.44
  - scipy>=1.11
  - statsmodels>=0.14
  - geopandas>=0.14
  - rasterio>=1.3
  - matplotlib>=3.8
  - pyarrow>=15.0
  - joblib>=1.3
```

Random seeds are fixed throughout. The learner tournament, the cross-validation
split, the bootstrap intervals, and the attribution background samples are all
seeded, so a rerun on the same data reproduces the same numbers.

---

## Reproducibility statement

Every number in the paper is generated by the code in this repository and stored
in `tables/`. The manuscript reads its numbers from a single LaTeX constants
file (`paper/*_constants.tex`) that is populated from those tables, so the text
and the results cannot disagree.

If you find a discrepancy, please
[open an issue](https://github.com/albertfaiz/how-long-not-how-much/issues) and we will resolve it.

---

## Citation

If you use this code or these results, please cite the paper:

```bibtex
@article{Ahmad2026Paper1,
  author  = {Ahmad, Faiz and Lary, David J. and Ruwali, Shisir and Shrestha, Samyak and Aker, Adam and Waczak, John and Madushanka, Prabuddha},
  title   = {How Long, Not How Much: Exposure Duration Outranks Concentration in an Eighteen-Modality Satellite and Reanalysis Fusion for United States County Life Expectancy},
  journal = {Sensors},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff). GitHub renders
it as a "Cite this repository" button.

---

## License

Code is released under the MIT License, see
[`LICENSE`](LICENSE). Figures, tables, and derived data are released under
CC-BY-4.0. Input datasets remain under the licenses of their original
providers, listed above.

---

## Contact

**Faiz Ahmad** (lead author)
faiz.ahmad@utdallas.edu &nbsp;|&nbsp; ORCID [0009-0006-0330-8993](https://orcid.org/0009-0006-0330-8993)

Department of Physics, The University of Texas at Dallas, 800 W Campbell Rd, Richardson, TX 75080, USA
[Multi-Scale Integrated Intelligent Interactive Sensing (MINTS)](https://mints.utdallas.edu)

For questions about reproducing the analysis, please open an issue rather than
emailing, so the answer is visible to others.

---

<div align="center">
<sub>Generated 2026-08-09 &nbsp;|&nbsp; version 1.0.0</sub>
</div>
